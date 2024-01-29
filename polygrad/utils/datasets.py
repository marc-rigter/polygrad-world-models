import numpy as np
import dill as pickle


def reload_dataset(reload_dataset_path, new_dataset):
    data = np.load(reload_dataset_path, allow_pickle=True).item()

    # put the loaded data into the dataset object
    new_dataset.reset_data_buffer()
    steps_reloaded = 0
    for i in range(data["path_lengths"].shape[0]):
        path_length = data["path_lengths"][i]
        episode = {
            "observations": data["observations"][i][:path_length],
            "actions": data["actions"][i][:path_length],
            "next_observations": data["next_observations"][i][:path_length],
            "rewards": data["rewards"][i][:path_length],
            "terminals": data["terminals"][i][:path_length],
            "sim_states": data["sim_states"][i][:path_length],
        }
        steps_reloaded += path_length
        episode["timeouts"] = np.array([False] * len(episode["rewards"]))
        new_dataset.add_episode(episode)
    new_dataset.update_normalizers()
    print(
        f"Loaded dataset containing {new_dataset.data_buffer.n_episodes} episodes and {steps_reloaded} steps."
    )
    return new_dataset
