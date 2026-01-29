from datasets import load_dataset


def load_xsum(split="train"):
    """
    XSum fields include: document, summary, id.
    """
    return load_dataset("xsum", split=split)


def iter_examples(ds, max_examples=None):
    n = 0
    for ex in ds:
        doc = ex.get("document", "")
        summ = ex.get("summary", "")
        ex_id = ex.get("id", None)

        if not doc or not summ:
            continue

        yield {"id": ex_id, "document": doc.strip(), "summary": summ.strip()}
        n += 1
        if max_examples is not None and n >= max_examples:
            break
