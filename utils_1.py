import re


def pickle_name_to_mask_name(pickle_name):
    video_id = re.findall(r'\d+', pickle_name)
    assert len(video_id) > 0
    video_id = video_id[0]
    scene_name = pickle_name[:pickle_name.find(video_id)-5]
    return scene_name + '_' + video_id + '_mask.png'

def get_goal_data(goal_data, pickle_name):
    goal_names = goal_data[2]
    assert pickle_name in goal_names
    return goal_data[1][goal_names.index(pickle_name)]

if __name__ == '__main__':
    print(pickle_name_to_mask_name('bookstorevideo4.pickle'))
