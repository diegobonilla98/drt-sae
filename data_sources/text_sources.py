from itertools import islice


def iter_hf_texts(dataset_name: str, dataset_config: str | None, split: str, text_column: str, limit: int | None = None):
    from datasets import load_dataset
    if dataset_config is None:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    iterator = dataset if limit is None else islice(dataset, limit)
    for row in iterator:
        text = row.get(text_column, "")
        if text is not None and len(str(text).strip()) > 0:
            yield str(text)


def batched_texts(iterator, batch_size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
