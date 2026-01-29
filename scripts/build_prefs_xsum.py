import argparse
import json
from tqdm import tqdm

from data.xsum_loader import load_xsum, iter_examples
from speaker.gpt2_speaker import GPT2Speaker
from listener.bertscore_listener import BERTScoreListener


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_rows_for_buffer(buffer, cand_lists, listener):
    """
    buffer: list of dicts {id, prompt, gold}
    cand_lists: list of list of strings
    listener: BERTScoreListener

    returns: list of output rows for DPO training
    """
    # flatten for bertsscore batching
    flat_cands = []
    flat_refs = []
    for b, cands in zip(buffer, cand_lists):
        for c in cands:
            flat_cands.append(c)
            flat_refs.append(b["gold"])

    scores = listener.score_f1_batch(flat_cands, flat_refs)

    out_rows = []
    s_idx = 0
    for b, cands in zip(buffer, cand_lists):
        cand_scores = scores[s_idx:s_idx + len(cands)]
        s_idx += len(cands)

        best_i = max(range(len(cand_scores)), key=lambda i: cand_scores[i])
        worst_i = min(range(len(cand_scores)), key=lambda i: cand_scores[i])

        chosen = cands[best_i]
        rejected = cands[worst_i]
        chosen_score = cand_scores[best_i]
        rejected_score = cand_scores[worst_i]

        # safety check, choosen should have higher score
        assert chosen_score >= rejected_score, {
            "cand_scores": cand_scores,
            "best_i": best_i,
            "worst_i": worst_i
        }

        out_rows.append({
            "id": b["id"],
            "prompt": b["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "meta": {
                "gold_summary": b["gold"],
                "candidates": cands,
                "scores": cand_scores,
                "chosen_idx": best_i,
                "rejected_idx": worst_i,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
            }
        })

    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_examples", type=int, default=2000)
    parser.add_argument("--outfile", required=True)

    # speaker
    parser.add_argument("--speaker_model", default="gpt2")
    parser.add_argument("--n_candidates", type=int, default=2)
    parser.add_argument("--decode_mode", default="sample", choices=["sample", "beam"])
    parser.add_argument("--max_doc_tokens", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--token_budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gen_batch_size", type=int, default=2)

    # listener
    parser.add_argument("--bertscore_model", default="roberta-base")
    parser.add_argument("--bertscore_batch_size", type=int, default=16)

    args = parser.parse_args()

    if args.n_candidates < 2:
        raise ValueError("--n_candidates must be >= 2 for DPO preferences")

    ds = load_xsum(split=args.split)

    speaker = GPT2Speaker(model_name=args.speaker_model)
    listener = BERTScoreListener(
        model_type=args.bertscore_model,
        batch_size=args.bertscore_batch_size
    )

    out_rows = []
    buffer = []
    idx = 0

    for ex in tqdm(iter_examples(ds, max_examples=args.max_examples), total=args.max_examples, desc="Building prefs"):
        # Truncate doc for prompt input length control
        doc = speaker.truncate_document(ex["document"], max_doc_tokens=args.max_doc_tokens)
        prompt = speaker.build_prompt(doc, token_budget=args.token_budget)
        gold = ex["summary"]

        buffer.append({
            "id": ex["id"],
            "prompt": prompt,
            "gold": gold
        })

        if len(buffer) < args.gen_batch_size:
            continue

        # generate candidates batch
        prompts = [b["prompt"] for b in buffer]
        cand_lists = speaker.generate_candidates_batch(
            prompts,
            n_candidates=args.n_candidates,
            max_new_tokens=args.max_new_tokens,
            mode=args.decode_mode,
            seed=args.seed + idx
        )

        rows = build_rows_for_buffer(buffer, cand_lists, listener)

        # add meta info
        for r in rows:
            r["meta"]["decode_mode"] = args.decode_mode
            r["meta"]["speaker_model"] = args.speaker_model
            r["meta"]["bertscore_model"] = args.bertscore_model
            r["meta"]["max_new_tokens"] = args.max_new_tokens
            r["meta"]["token_budget"] = args.token_budget

        out_rows.extend(rows)
        idx += len(rows)
        buffer = []

    # clean up remaining buffer
    if buffer:
        prompts = [b["prompt"] for b in buffer]
        cand_lists = speaker.generate_candidates_batch(
            prompts,
            n_candidates=args.n_candidates,
            max_new_tokens=args.max_new_tokens,
            mode=args.decode_mode,
            seed=args.seed + idx
        )

        rows = build_rows_for_buffer(buffer, cand_lists, listener)
        for r in rows:
            r["meta"]["decode_mode"] = args.decode_mode
            r["meta"]["speaker_model"] = args.speaker_model
            r["meta"]["bertscore_model"] = args.bertscore_model
            r["meta"]["max_new_tokens"] = args.max_new_tokens
            r["meta"]["token_budget"] = args.token_budget

        out_rows.extend(rows)

    write_jsonl(args.outfile, out_rows)
    print("Wrote {} preference examples to {}".format(len(out_rows), args.outfile))


if __name__ == "__main__":
    main()
