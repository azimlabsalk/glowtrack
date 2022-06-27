

def add_series():
    """Add a clip series to the database.

    A series is a directory that contains a bunch of clipgroups: clip0/ ... clipN/

    A series should contain a file info.yaml. Example:

    # info.yaml
    mouse: gtacr-8
    landmark: left-hindpaw
    every_nth_pair: 1

    """
    pass

# def add_dataset(clip_paths, clipset_name, session, make_set=False,
#                      strobed=True):
# 
#     if make_set:
#         clipset = ClipSet(name=clipset_name)
#         session.add(clipset)
#         session.commit()
#     else:
#         clipset = session.query(ClipSet).filter_by(name=clipset_name).one()
# 
#     new_clips = []
#     for clip_path in clip_paths:
#         print('adding path {} to clipset {}'.format(
#             clip_path, clipset_name))
# 
#         new_clip = Clip(path=clip_path, strobed=strobed)
#         new_clips.append(new_clip)
# 
#     for new_clip in new_clips:
#         session.add(new_clip)
#         session.commit()
# 
#     clipset.clips.extend(new_clips)
#     session.add(clipset)
#     session.commit()


if __name__ == '__main__':
    print('adding datasets')

