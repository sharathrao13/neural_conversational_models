import data_preprocessor as dp

def read_data(source_path, target_path, _buckets):
    data_set = [[] for _ in _buckets]
    with open(source_path, mode="r") as source_file:
        with open(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            while source and target:

                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(dp.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set