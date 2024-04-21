import copy
import itertools

def pre_process_landmark(landmark_list):
    # Create a deep copy of the input landmark list to avoid modifying the original list
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Initialize base coordinates with the first landmark point
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        # Subtract the base coordinates from each landmark point
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Flatten the 2D list to a 1D list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Find the maximum absolute value in the 1D list
    max_value = max(list(map(abs, temp_landmark_list)))

    # Normalize each value in the 1D list by dividing it by the maximum absolute value
    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
