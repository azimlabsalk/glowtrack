import os

from sqlalchemy import (Boolean, Column, ForeignKey, Float, Index, Integer,
                        MetaData, String, Table)
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import object_session, relationship
from sqlalchemy.orm.exc import NoResultFound

from yogi.utils import contains_duplicates

meta = MetaData(naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
      })

Base = declarative_base(metadata=meta)


clipset_association_table = Table(
    'clipset_association', Base.metadata,
    Column('clip_id', Integer, ForeignKey('clips.id'), index=True),
    Column('clipset_id', Integer, ForeignKey('clipsets.id'), index=True)
)


imageset_association_table = Table(
    'imageset_association', Base.metadata,
    Column('image_id', Integer, ForeignKey('images.id'), index=True),
    Column('imageset_id', Integer, ForeignKey('imagesets.id'), index=True),
    Index('imageset_joint_idx', 'image_id', 'imageset_id')
)

subclipset_association_table = Table(
    'subclipset_association', Base.metadata,
    Column('subclip_id', Integer, ForeignKey('subclips.id'), index=True),
    Column('subclipset_id', Integer, ForeignKey('subclipsets.id'), index=True),
    Index('subclipset_joint_idx', 'subclip_id', 'subclipset_id')
)

labelset_association_table = Table(
    'labelset_association', Base.metadata,
    Column('label_id', Integer, ForeignKey('labels.id'), index=True),
    Column('labelset_id', Integer, ForeignKey('labelsets.id'), index=True),
    Index('labelset_joint_idx', 'label_id', 'labelset_id')
)

clipgroupset_association_table = Table(
    'clipgroupset_association', Base.metadata,
    Column('clipgroup_id', Integer, ForeignKey('clip_groups.id'), index=True),
    Column('clipgroupset_id', Integer, ForeignKey('clipgroupsets.id'),
           index=True),
    Index('clipgroupset_joint_idx', 'clipgroup_id', 'clipgroupset_id')
)


landmarkset_association_table = Table(
    'landmarkset_association', Base.metadata,
    Column('landmark_id', Integer, ForeignKey('landmarks.id'), index=True),
    Column('landmarkset_id', Integer, ForeignKey('landmarksets.id'),
           index=True),
    Index('landmarkset_joint_idx', 'landmark_id', 'landmarkset_id')
)


class Clip(Base):
    __tablename__ = 'clips'
    id = Column(Integer, primary_key=True)
    path = Column(String())
    color_type = Column(String())
    camera_index = Column(Integer)
    strobed = Column(Boolean)
    flipped = Column(Boolean)
    uv_first = Column(Boolean)
    clip_group_id = Column(Integer, ForeignKey('clip_groups.id'), index=True)

    mouse_digit_length = Column(Float)

    height = Column(Integer)
    width = Column(Integer)

    frame_rate = Column(Integer)

    images = relationship("Image", backref="clip",
                          cascade="all, delete-orphan")

    @classmethod
    def display_attrs(self):
        col_names = [col.name for col in self.__table__.columns]
        attrs = col_names + ['num_frames']
        return attrs

    def __init__(self, **kwargs):
        assert('width' not in kwargs)
        assert('height' not in kwargs)
        try:
            kwargs['path'] = os.path.realpath(kwargs['path'])
        except KeyError:
            pass
        super().__init__(**kwargs)
        self.set_size_from_metadata()

    def __repr__(self):
        return ('<Clip id="{}", length="{}", path="{}", height="{}", '
                'width="{}", color_type="{}">'.format(
                    self.id, self.num_frames, self.path, self.height,
                    self.width, self.color_type))

    def to_dict(self):
        data = {
            'id': self.id,
            'path': self.path,
            'height': self.height,
            'width': self.width,
        }
        return data

    def set_size_from_metadata(self):
        from yogi.video import get_video_size
        video_path = self.video_path('visible')
        (self.width, self.height) = get_video_size(video_path)

    def video_path(self, illumination):
        if self.strobed:
            return os.path.join(self.path, illumination, 'video.mp4')
        else:
            assert(illumination != 'uv')
            return self.path

    def frames_path(self, illumination, dir_name=None):
        if dir_name is None:
            dir_name = 'frames'

        video_path = self.video_path(illumination)
        video_dir = os.path.dirname(video_path)

        if self.strobed:
            frames_dir = os.path.join(video_dir, dir_name)
        else:
            assert(illumination != 'uv')
            base_path = os.path.splitext(video_path)[0]
            frames_dir = os.path.join(base_path, dir_name)

        return frames_dir

    def video_to_frames(self, session, dir_name='frames',
                        illumination='visible', overwrite=False):
        from yogi.video import video_to_frames

        if not self.has_frames(illumination):

            video_path = self.video_path(illumination)

            # make frames directory
            frames_dir = self.frames_path(illumination, dir_name=dir_name)
            os.makedirs(frames_dir, exist_ok=overwrite)

            frame_paths = video_to_frames(video_path, frames_dir,
                                          flip=self.flipped)

            if len(frame_paths) == 0:
                return

            for i, frame_path in enumerate(frame_paths):
                image = Image(path=frame_path, clip_id=self.id, frame_num=i,
                              height=self.height, width=self.width,
                              illumination=illumination)
                session.add(image)
            session.commit()

    def has_frames(self, illumination):
        session = object_session(self)
        exists = session.query(
            session.query(Image).filter_by(clip_id=self.id,
                                           illumination=illumination).exists()
            ).scalar()
        return exists

    def compute_centroids(self, session, dye_detector_name, batch_size=10):
        import imageio

        detector = session.query(DyeDetector).filter_by(
            name=dye_detector_name).one()

        uv_video_path = self.video_path(illumination='uv')
        uv_reader = imageio.get_reader(uv_video_path)

        count = 0
        for i, uv_image in enumerate(uv_reader.iter_data()):

            image = session.query(Image).filter_by(
                clip_id=self.id, frame_num=i, illumination='visible').first()

            if image is None:
                print('clip {} has not been converted to images'.format(
                    self.id))
                print(('You probably need to run '
                       '\'yogi video-to-frames clip {}\''.format(self.id)))
                return

            (x, y, area, mean_value) = detector.apply(uv_image)
            label = Label(image_id=image.id, x=x, y=y, source_id=detector.id,
                          area=float(area), mean_value=float(mean_value))
            session.add(label)

            count += 1
            if count % batch_size == 0:
                session.commit()

        session.commit()

    @property
    def num_frames(self):
        session = object_session(self)
        n = session.query(Image).\
            filter_by(clip_id=self.id, illumination='visible').count()
        return n

    @property
    def ordered_frames(self):
        session = object_session(self)
        images = session.query(Image)\
            .filter(Image.clip_id == self.id)\
            .filter(Image.illumination == 'visible')\
            .order_by(Image.frame_num.asc())\
            .all()
        return images

    @staticmethod
    def create_clips(clip_paths, clipset_name, session, make_set=False,
                     strobed=True, flipped=False):

        if make_set:
            clipset = ClipSet(name=clipset_name)
            session.add(clipset)
            session.commit()
        else:
            clipset = session.query(ClipSet).filter_by(name=clipset_name).one()

        new_clips = []
        for clip_path in clip_paths:
            print('adding path {} to clipset {}'.format(
                clip_path, clipset_name))

            new_clip = Clip(path=clip_path, strobed=strobed, flipped=flipped)
            new_clips.append(new_clip)
            print('clip created')

        for new_clip in new_clips:
            print('committing new clip')
            session.add(new_clip)
            session.commit()

        clipset.clips.extend(new_clips)
        session.add(clipset)
        session.commit()

    def get_labels(self, session, label_source_name, return_array=False):

        label_source = session.query(LabelSource).filter_by(
            name=label_source_name).one()

        labels = session.query(Label)\
            .filter(Label.source_id == label_source.id)\
            .join(Image)\
            .filter(Image.clip_id == self.id)\
            .filter(Label.image_id == Image.id)\
            .order_by(Image.frame_num.asc())\
            .all()

        if return_array:
            import numpy as np
            labels = np.array([(label.x, label.y, label.confidence)
                               for label in labels])

        return labels

    def has_labels_from(self, label_source, landmark_id=None):

        session = object_session(self)
        query = session.query(Image)\
            .join(Label)\
            .filter(Image.id == Label.image_id)\
            .filter(Label.source_id == label_source.id)\
            .filter(Image.clip_id == self.id)

        if landmark_id is not None:
            query = query.filter(Label.landmark_id == landmark_id)

        labeled_images = query.all()

        labeled_image_ids = [image.id for image in labeled_images]
        image_ids = [image.id for image in self.images]

        has_labels = (set(image_ids) <= set(labeled_image_ids))
        return has_labels

    def get_unlabeled_images(self, session, source_name):
        labeled_images = set(self.get_labeled_images(session, source_name))
        all_images = set(self.images)
        unlabeled_images = all_images - labeled_images
        return unlabeled_images

    def get_labeled_images(self, session, source_name):
        label_source = session.query(LabelSource).filter_by(
            name=source_name).one()
        images = session.query(Image)\
                        .join(Label)\
                        .filter(Label.source_id == label_source.id)\
                        .filter(Label.image_id == Image.id)\
                        .filter(Image.clip_id == self.id).all()
        return images


class ClipGroup(Base):
    __tablename__ = 'clip_groups'
    id = Column(Integer, primary_key=True)
    path = Column(String())
    clips = relationship('Clip')


class ClipGroupSet(Base):
    __tablename__ = 'clipgroupsets'
    id = Column(Integer, primary_key=True)
    name = Column(String())
    clipgroups = relationship("ClipGroup",
                              secondary=clipgroupset_association_table)


class CharucoBoard(Base):
    __tablename__ = 'charuco_boards'
    id = Column(Integer, primary_key=True)
    name = Column(String())
    number_rows = Column(Integer)
    number_cols = Column(Integer)
    square_length = Column(Float)
    marker_length = Column(Float)
    marker_bits = Column(Integer)
    dict_size = Column(Integer)


class Calibration(Base):
    __tablename__ = 'calibrations'
    id = Column(Integer, primary_key=True)
    name = Column(String())
    path = Column(String())
    videos = relationship('CalibrationVideo')
    charuco_board_id = Column(Integer, ForeignKey('charuco_boards.id'))
    charuco_board = relationship('CharucoBoard',
                                 foreign_keys=[charuco_board_id])


class CalibrationVideo(Base):
    __tablename__ = 'calibration_videos'
    id = Column(Integer, primary_key=True)
    path = Column(String())
    camera_index = Column(Integer)
    calibration_id = Column(Integer, ForeignKey('calibrations.id'))


class Reconstruction(Base):
    __tablename__ = 'reconstructions'
    id = Column(Integer, primary_key=True)
    calibration_id = Column(Integer, ForeignKey('calibrations.id'))
    calibration = relationship('Calibration')
    clipgroupset_id = Column(Integer, ForeignKey('clipgroupsets.id'))
    clipgroupset = relationship('ClipGroupSet')
    labelset_id = Column(Integer, ForeignKey('labelsets.id'))
    labelset = relationship('LabelSet')


class Point(Base):
    __tablename__ = 'points'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    reconstruction_id = Column(Integer, ForeignKey('reconstructions.id'))
    clip_group_id = Column(Integer, ForeignKey('clip_groups.id'))
    frame_num = Column(Integer)


class SubClip(Base):
    __tablename__ = 'subclips'
    id = Column(Integer, primary_key=True)
    clip_id = Column(Integer, ForeignKey('clips.id'), index=True)
    start_idx = Column(Integer)
    end_idx = Column(Integer)

    # clip.has_labels_from
    # clip.get_unlabeled_images(session, self.name)

    clip = relationship('Clip', lazy='select')

    def __repr__(self):
        return ('<SubClip id="{}", clip_id="{}", start_idx="{}", '
                'end_idx="{}", size="{}">'.format(
                    self.id, self.clip_id, self.start_idx, self.end_idx,
                    self.size))

    def to_dict(self):
        data = {
            'id': self.id,
            'path': self.clip.path,
            'height': self.clip.height,
            'width': self.clip.width,
        }
        return data

    def has_labels_from(self, label_source, landmark_id=None):

        session = object_session(self)
        query = session.query(Image)\
            .join(Label)\
            .filter(Image.id == Label.image_id)\
            .filter(Label.source_id == label_source.id)\
            .filter(Image.clip_id == self.clip_id)

        if landmark_id is not None:
            query = query.filter(Label.landmark_id == landmark_id)

        labeled_images = query.all()

        labeled_image_ids = [image.id for image in labeled_images]
        image_ids = [image.id for image in self.images]

        has_labels = (set(image_ids) <= set(labeled_image_ids))
        return has_labels

    def get_unlabeled_images(self, session, source_name, landmark_id=None):
        labeled_images = set(self.get_labeled_images(session, source_name,
                                                     landmark_id=landmark_id))
        all_images = set(self.images)
        unlabeled_images = list(all_images - labeled_images)
        return unlabeled_images

    def get_labeled_images(self, session, source_name, landmark_id=None):
        label_source = session.query(LabelSource).filter_by(
            name=source_name).one()
        query = session.query(Image)\
                       .join(Label)\
                       .filter(Label.source_id == label_source.id)\
                       .filter(Label.image_id == Image.id)\
                       .filter(Image.clip_id == self.clip_id)

        if landmark_id is not None:
            query = query.filter(Label.landmark_id == landmark_id)

        labeled_images = query.all()

        return labeled_images

    @property
    def size(self):
        return len(self.images)

    @property
    def images(self):
        return self.clip.ordered_frames[self.start_idx:self.end_idx]

    @property
    def ordered_frames(self):
        return self.images


class SubClipSet(Base):
    __tablename__ = 'subclipsets'
    id = Column(Integer, primary_key=True)
    subclips = relationship("SubClip", secondary=subclipset_association_table)
    name = Column(String(), unique=True)
    ordered_subclips = relationship("SubClip",
                             secondary=subclipset_association_table,
                             order_by=(subclipset_association_table.c.
                                       subclip_id))

    def __repr__(self):
        return '<SubClipSet id="{}", name="{}">, size="{}">'.format(
            self.id, self.name, self.size)

    def to_dict(self):
        data = {
            'id': self.id,
            'name': self.name,
            'length': self.size,
        }
        return data

    @classmethod
    def display_attrs(self):
        col_names = [col.name for col in self.__table__.columns]
        attrs = col_names + ['size']
        return attrs

    @property
    def size(self):
        session = object_session(self)
        count = session.query(subclipset_association_table).filter_by(
            subclipset_id=self.id).count()
        return count

    @property
    def images(self):
        images = []
        for subclip in self.subclips:
            images.extend(subclip.images)
        return images

    @property
    def num_images(self):
        session = object_session(self)
        count = session.query(Image, SubClip)\
             .filter(Image.clip_id == SubClip.clip_id)\
             .join(subclipset_association_table)\
             .join(SubClipSet)\
             .filter(SubClipSet.id == self.id)\
             .filter(or_(Image.frame_num >= SubClip.start_idx,
                         SubClip.start_idx == None))\
             .filter(or_(Image.frame_num < SubClip.end_idx,
                         SubClip.end_idx == None))\
             .filter(Image.illumination == 'visible')\
             .count()
        return count

    @property
    def label_source_ids(self):
        session = object_session(self)

        source_label_counts = \
            session.query(Label.source_id, func.count(Label.source_id)).\
            select_from(Image, Label).\
            join(SubClipSet.subclips).\
            filter(SubClip.clip_id == Image.clip_id).\
            filter(Image.id == Label.image_id).\
            filter(SubClipSet.id == self.id).\
            filter(or_(Image.frame_num >= SubClip.start_idx,
                        SubClip.start_idx == None)).\
            filter(or_(Image.frame_num < SubClip.end_idx,
                        SubClip.end_idx == None)).\
            group_by(Label.source_id).all()

        image_count = self.num_images
        source_ids = [source_id for (source_id, label_count) in
                      source_label_counts if label_count == image_count]
        return source_ids



class ClipSet(Base):
    __tablename__ = 'clipsets'
    id = Column(Integer, primary_key=True)
    name = Column(String(), unique=True)
    clips = relationship("Clip", secondary=clipset_association_table,
                         backref="clipsets")
    ordered_clips = relationship("Clip",
                             secondary=clipset_association_table,
                             order_by=(clipset_association_table.c.
                                       clip_id))


    def __repr__(self):
        return '<ClipSet id="{}", name="{}", len="{}">'.format(
            self.id, self.name, len(self.clips))

    @classmethod
    def display_attrs(self):
        col_names = [col.name for col in self.__table__.columns]
        attrs = col_names + ['num_clips', 'num_images']
        return attrs

    @property
    def num_clips(self):
        return len(self.clips)

    @property
    def num_images(self):
        session = object_session(self)
        count = session.query(Image)\
            .join(Clip)\
            .join(clipset_association_table)\
            .join(ClipSet)\
            .filter(ClipSet.id == self.id)\
            .count()
        return count

    @property
    def label_source_ids(self):
        session = object_session(self)

        source_label_counts = \
            session.query(Label.source_id, func.count(Label.source_id)).\
            select_from(Image, Label).\
            join(ClipSet.clips).\
            filter(Clip.id == Image.clip_id).\
            filter(Image.id == Label.image_id).\
            filter(ClipSet.id == self.id).\
            group_by(Label.source_id).all()
        image_count = self.num_images
        source_ids = [source_id for (source_id, label_count) in
                      source_label_counts if label_count == image_count]
        return source_ids

    def to_dict(self):
        data = {
            'id': self.id,
            'name': self.name,
            'length': self.num_clips,
        }
        return data

    def get_images(self, session):
        images = session.query(Image).\
                         join(Clip).\
                         join(clipset_association_table).\
                         join(ClipSet).\
                         filter(ClipSet.id == self.id).all()
        return images


class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    height = Column(Integer)
    width = Column(Integer)
    clip_id = Column(Integer, ForeignKey('clips.id'), index=True)
    frame_num = Column(Integer)
    path = Column(String(), index=True)
    illumination = Column(String())
    light_index = Column(Integer)

    bounding_boxes = relationship("BoundingBox", backref="image")

    def to_dict(self):
        data = {
            'width': self.width,
            'height': self.height,
            'id': self.id,
        }
        return data

    def has_label_from(self, label_source_id, session):
        exists = session.query(
            session.query(Label).filter_by(image_id=self.id,
                                           source_id=label_source_id).exists()
            ).scalar()
        return exists

    def get_array(self):
        from scipy.misc import imread
        from skimage.color import gray2rgb
        img = imread(self.path)
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = gray2rgb(img)
        return img

    def render_labels(self, label_source_ids, session, conf_threshold=None,
                      landmark_id=None, landmarkset_id=None, color=None, show_conf=True):
        from yogi.graphics import render_labels

        assert((landmark_id is None) or (landmarkset_id is None))

        sources_labels = []
        for label_source_id in label_source_ids:

            query = session.query(Label)\
                .filter(Label.image_id == self.id)\
                .filter(Label.source_id == label_source_id)\
                .order_by(Label.landmark_id.asc())

            if landmark_id is not None:
                query = query.filter(Label.landmark_id == landmark_id)

            if landmarkset_id is not None:
                query = query.join(Landmark)\
                    .filter(Label.landmark_id == Landmark.id)\
                    .join(landmarkset_association_table)\
                    .join(LandmarkSet)\
                    .filter(LandmarkSet.id == landmarkset_id)\

            source_labels = query.all()
            sources_labels.append(source_labels)

        # for now, just put all the labels together
        labels = [label for source_labels in sources_labels
                  for label in source_labels]

        img = self.get_array()
        img = render_labels(img, labels, conf_threshold=conf_threshold,
                            color=color, show_conf=show_conf)

        return img

    def __repr__(self):
        s = ('<Image id="{}", clip_id="{}", frame_num="{}", path="{}", '
             'height="{}", width="{}">').format(
             self.id, self.clip_id, self.frame_num, self.path, self.height,
             self.width)
        return s


class ImageSet(Base):
    __tablename__ = 'imagesets'
    id = Column(Integer, primary_key=True)
    images = relationship('Image', secondary=imageset_association_table)
    name = Column(String(), unique=True)

    def __repr__(self):
        return '<ImageSet id="{}", name="{}">, len="{}">'.format(
            self.id, self.name, self.num_images)

    def get_labels(self, session, label_source_name):
        label_source = session.query(LabelSource).filter_by(
            name=label_source_name).one()
        labels = session.query(Label)\
                        .filter(Label.source_id == label_source.id)\
                        .join(Image)\
                        .filter(Label.image_id == Image.id)\
                        .join(imageset_association_table)\
                        .join(ImageSet)\
                        .filter(ImageSet.id == self.id).all()
        return labels

    def get_unlabeled_images(self, session, source_name):
        labeled_images = set(self.get_labeled_images(session, source_name))
        all_images = set(self.images)
        unlabeled_images = all_images - labeled_images
        return unlabeled_images

    def get_labeled_images(self, session, source_name):
        label_source = session.query(LabelSource).filter_by(
            name=source_name).one()
        images = session.query(Image)\
                        .join(Label)\
                        .filter(Label.source_id == label_source.id)\
                        .filter(Label.image_id == Image.id)\
                        .join(imageset_association_table)\
                        .join(ImageSet)\
                        .filter(ImageSet.id == self.id).all()
        return images

    @classmethod
    def display_attrs(self):
        col_names = [col.name for col in self.__table__.columns]
        attrs = col_names + ['num_images']
        return attrs

    @property
    def label_source_ids(self):
        session = object_session(self)

        source_label_counts = \
            session.query(Label.source_id, func.count(Label.source_id)).\
            select_from(Image, Label).\
            join(ImageSet.images).\
            filter(ImageSet.id == self.id).\
            filter(Image.id == Label.image_id).\
            group_by(Label.source_id).all()

        image_count = self.num_images
        source_ids = [source_id for (source_id, label_count) in
                      source_label_counts if label_count == image_count]

        return source_ids

    @property
    def label_sources(self):
        session = object_session(self)

        label_sources = []
        for id in self.label_source_ids:
            label_source = session.query(LabelSource).filter_by(id=id).one()
            label_sources.append(label_source)

        return label_sources

    @property
    def num_images(self):
        session = object_session(self)
        count = session.query(func.count()).\
            select_from(Image).\
            join(ImageSet.images).\
            filter(ImageSet.id == self.id).scalar()
        return count


class Label(Base):
    __tablename__ = 'labels'
    id = Column(Integer, primary_key=True)

    image_id = Column(Integer, ForeignKey('images.id'), index=True)
    image = relationship('Image', lazy='select')

    source_id = Column(Integer, ForeignKey('label_sources.id'),
                       index=True)

    landmark_id = Column(Integer, ForeignKey('landmarks.id'))
    landmark = relationship('Landmark', lazy='select')

    x = Column(Float)
    y = Column(Float)
    confidence = Column(Float)
    scale = Column(Float)
    occluded = Column(Boolean)

    area = Column(Float)
    mean_value = Column(Float)

    def is_hidden(self):
        return (self.x is None)

    def to_dict(self):
        data = {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'confidence': self.confidence,
            'image_id': self.image_id,
            'landmark_id': self.landmark_id,
            'source_id': self.source_id,
        }
        return data

    @property
    def x_px(self):
        return (None if self.is_hidden() else (self.x * self.image.width))

    @property
    def y_px(self):
        return (None if self.is_hidden() else (self.y * self.image.height))

    @property
    def x_digit(self):
        return (None if self.is_hidden() else (self.x * self.image.width /
                                               self.image.clip
                                               .mouse_digit_length))

    @property
    def y_digit(self):
        return (None if self.is_hidden() else (self.y * self.image.height /
                                               self.image.clip
                                               .mouse_digit_length))

    @property
    def x_norm(self):
        return self.x

    @property
    def y_norm(self):
        return self.y

    @staticmethod
    def labels_for_imageset(session, label_source, imageset):
        labels = session.query(Label)\
            .join(Image)\
            .filter(Label.source_id == label_source.id)\
            .filter(Label.image_id == Image.id)\
            .join(imageset_association_table)\
            .join(ImageSet)\
            .filter(ImageSet.id == imageset.id).all()
        return labels

    def __repr__(self):
        return ('<Label id="{}", image_id="{}", source_id="{}",'
                ' x="{}", y="{}", confidence="{}">').format(
                    self.id, self.image_id, self.source_id, self.x, self.y,
                    self.confidence)


class LabelSet(Base):
    __tablename__ = 'labelsets'
    id = Column(Integer, primary_key=True)
    labels = relationship('Label', secondary=labelset_association_table)
    name = Column(String(), unique=True)

    def to_raw_python(self, session):

        labeled_images = session.query(Image, Label)\
                                .filter(Label.image_id == Image.id)\
                                .join(labelset_association_table)\
                                .join(LabelSet)\
                                .filter(LabelSet.id == self.id).all()

        def replace_none(x):
            return x if x is not None else 'hidden'

        labelset = []
        for (image, label) in labeled_images:
            (w, h) = (image.width, image.height)
            x = replace_none(label.x)
            y = replace_none(label.y)
            labelset.append((image.path, w, h, x, y))

        # sanity check
        image_paths = [row[0] for row in labelset]
        if contains_duplicates(image_paths):
            raise Exception("Multiple labels per image not yet supported.")

        return labelset

    @classmethod
    def display_attrs(self):
        col_names = [col.name for col in self.__table__.columns]
        attrs = col_names + ['num_labels']
        return attrs

    @property
    def num_labels(self):
        session = object_session(self)
        count = session.query(func.count()).\
            select_from(Label).\
            join(LabelSet.labels).\
            filter(LabelSet.id == self.id).scalar()
        return count

    def __repr__(self):
        return '<LabelSet id="{}", name="{}">, len="{}">'.format(
            self.id, self.name, self.num_labels)


class LabelSource(Base):
    __tablename__ = 'label_sources'
    id = Column(Integer, primary_key=True)
    name = Column(String(), unique=True)
    source_type = Column(String(50))

    __mapper_args__ = {
        'polymorphic_identity': 'labelsource',
        'polymorphic_on': source_type,
    }


class Smoother(Base):
    """Class responsible for smoothing time series stored in numpy format.

    Subclasses should implement 'smooth(self, label_array)' with arguments:

    label_array - N x 3 numpy array, with columns (x, y, confidence)
    """
    __tablename__ = 'smoothers'
    id = Column(Integer, primary_key=True)
    smoother_type = Column(String(50))
    name = Column(String(), unique=True)

    def smooth(self, label_array):
        """Smoothes a time series.

        input - N x 3 numpy array, with columns (x, y, confidence)
        """
        raise NotImplementedError('user should override smooth()')

    def smooth_clipset(self, session, clipset_name, label_source_name):
        clipset = session.query(ClipSet).filter_by(name=clipset_name).one()

        label_source = session.query(LabelSource).filter_by(
            name=label_source_name).one()
        source_id = label_source.id
        smoothed_source = SmoothedLabelSource.find_or_create(
            session, source_id, self.id)

        landmarkset = label_source.landmarkset

        for landmark in landmarkset.landmarks:
            for clip in clipset.clips:
                print('smoothing: ' + clip.path)
                if not clip.has_labels_from(smoothed_source,
                                            landmark_id=landmark.id):
                    self.smooth_clip(session, clip.id, label_source_name,
                                     landmark_id=landmark.id)

    def smooth_clip(self, session, clip_id, label_source_name,
                    landmark_id=None):
        from yogi.sql import get_labels

        # do processing
        clip = session.query(Clip).filter_by(id=clip_id).one()
        label_array = get_labels(label_source_name,
                                 landmark_id=landmark_id,
                                 clip_id=clip.id, return_array=True)
        processed_array = self.smooth(label_array)
        assert(processed_array.shape[0] == label_array.shape[0])

        # save to db
        label_source = session.query(LabelSource).filter_by(
            name=label_source_name).one()
        source_id = label_source.id
        smoothed_source = SmoothedLabelSource.find_or_create(
            session, source_id, self.id)

        images = sorted(clip.images, key=lambda image: image.frame_num)
        for (row, image) in zip(processed_array, images):
            (x, y, confidence) = row
            label = Label(x=x, y=y, confidence=confidence, image_id=image.id,
                          source_id=smoothed_source.id,
                          landmark_id=landmark_id)
            session.add(label)

        session.commit()

    def smooth_subclipset(self, session, subclipset_name, label_source_name):
        subclipset = session.query(SubClipSet).filter_by(name=subclipset_name).one()

        label_source = session.query(LabelSource).filter_by(
            name=label_source_name).one()
        source_id = label_source.id
        smoothed_source = SmoothedLabelSource.find_or_create(
            session, source_id, self.id)

        landmarkset = label_source.landmarkset

        for landmark in landmarkset.landmarks:
            for subclip in subclipset.subclips:
                print('smoothing: ' + subclip.clip.path)
                if not subclip.has_labels_from(smoothed_source,
                                            landmark_id=landmark.id):
                    self.smooth_subclip(session, subclip.id, label_source_name,
                                     landmark_id=landmark.id)

    def smooth_subclip(self, session, subclip_id, label_source_name,
                    landmark_id=None):
        import numpy as np
        from yogi.sql import get_labels

        # do processing
        subclip = session.query(SubClip).filter_by(id=subclip_id).one()
        labels = get_labels(label_source_name,
                            landmark_id=landmark_id,
                            clip_id=subclip.clip_id)

        subclip_labels = []
        for label in labels:
            lower_bound_ok = subclip.start_idx is None or label.image.frame_num >= subclip.start_idx
            upper_bound_ok = subclip.end_idx is None or label.image.frame_num < subclip.end_idx
            if lower_bound_ok and upper_bound_ok:
                subclip_labels.append(label)

        label_array = np.array([(label.x, label.y, label.confidence)
                                for label in subclip_labels])

        print('label_array.shape[0] = {}'.format(label_array.shape[0]))
        print('subclip.size = {}'.format(subclip.size))
        assert(label_array.shape[0] == subclip.size)

        processed_array = self.smooth(label_array)
        assert(processed_array.shape[0] == label_array.shape[0])

        # save to db
        label_source = session.query(LabelSource).filter_by(
            name=label_source_name).one()
        source_id = label_source.id
        smoothed_source = SmoothedLabelSource.find_or_create(
            session, source_id, self.id)

        images = sorted(subclip.ordered_frames, key=lambda image: image.frame_num)
        for (row, image) in zip(processed_array, images):
            (x, y, confidence) = row
            label = Label(x=x, y=y, confidence=confidence, image_id=image.id,
                          source_id=smoothed_source.id,
                          landmark_id=landmark_id)
            session.add(label)

        session.commit()

    __mapper_args__ = {
        'polymorphic_identity': 'smoother',
        'polymorphic_on': smoother_type,
    }


class DyeSmoother(Smoother):
    __tablename__ = 'dye_smoothers'
    id = Column(Integer, ForeignKey('smoothers.id'), primary_key=True)
    type = Column(String(50))

    __mapper_args__ = {
        'polymorphic_identity': 'dye_smoother',
    }


class AdjMeanSmoother(Smoother):
    __tablename__ = 'adj_mean_smoothers'
    id = Column(Integer, ForeignKey('smoothers.id'), primary_key=True)
    tp_threshold = Column(Float)

    def smooth(self, label_array):
        """Smoothes a time series.

        label_array - N x 3 numpy array, with columns (x, y, confidence)
        """
        from yogi.post_processing_utils import adjacent_mean_smoother
        output = adjacent_mean_smoother(label_array,
                                        tp_threshold=self.tp_threshold)
        return output

    __mapper_args__ = {
        'polymorphic_identity': 'adj_mean',
    }


class MedianSmoother(Smoother):
    __tablename__ = 'median_smoothers'
    id = Column(Integer, ForeignKey('smoothers.id'), primary_key=True)
    kernel_size = Column(Integer)

    def smooth(self, label_array):
        """Smoothes a time series.

        label_array - N x 3 numpy array, with columns (x, y, confidence)
        """
        from yogi.post_processing_utils import median_filter
        output = median_filter(label_array, kernel_size=self.kernel_size)
        return output

    __mapper_args__ = {
        'polymorphic_identity': 'median',
    }


class CNNSmoother(Smoother):
    __tablename__ = 'cnn_smoothers'
    id = Column(Integer, ForeignKey('smoothers.id'), primary_key=True)

    path = Column(String())

    smoothing_factor = Column(Float)
    anomaly_threshold = Column(Float)
    block_length = Column(Integer)
    scale_factor = Column(Integer)

    def __init__(self, **kwargs):
        defaults = {
            'smoothing_factor': 0.005,
            'anomaly_threshold': 0.3,
            'block_length': 128,
            'scale_factor': 2,
        }
        kwargs = {**defaults, **kwargs}
        super().__init__(**kwargs)

    def smooth(self, label_array):
        """Smoothes a time series.

        label_array - N x 3 numpy array, with columns (x, y, confidence)
        """
        import os
        import torch
        import numpy as np
        from yogi.config import yogi_dir
        from yogi.smoothing import nn_smooth, UNet

        model = UNet(scale_factor=self.scale_factor)
        path = os.path.join(yogi_dir, self.path)
        model.load_state_dict(torch.load(path))
        model.eval()

        xy = nn_smooth(label_array[:, 0:2].T, model,
                       anomaly_thresh=self.anomaly_threshold,
                       smoothing_factor=self.smoothing_factor,
                       block_length=self.block_length).T

        conf = label_array[:, 2:3]
        output = np.concatenate((xy, conf), axis=1)
        return output

    __mapper_args__ = {
        'polymorphic_identity': 'cnn_smoother',
    }


class CorrectedLabelSource(LabelSource):
    __tablename__ = 'corrected_label_sources'
    id = Column(Integer, ForeignKey('label_sources.id'), primary_key=True)

    original_source_id = Column(Integer, ForeignKey('label_sources.id'),
                                index=True)
    original_source = relationship('LabelSource',
                                   foreign_keys=[original_source_id])

    __mapper_args__ = {
        'polymorphic_identity': 'corrected_labelsource',
        'inherit_condition': (id == LabelSource.id)
    }

    @staticmethod
    def create(original_source_id, session):
        original_source = session.query(LabelSource).filter_by(
            id=original_source_id).one()

        new_name = original_source.name + '-corrected'
        new_source = CorrectedLabelSource(
            original_source_id=original_source_id,
            name=new_name)

        session.add(new_source)
        session.commit()


def create_or_update_correction(image_id, original_source_id, x, y, session):

    source = get_corrected_source(original_source_id, session)

    if source is None:
        source = CorrectedLabelSource.create(original_source_id, session)

    corrected_labels = session.query(Label).filter_by(
        image_id=image_id,
        source_id=source.id).all()

    if len(corrected_labels) >= 2:
        # too many labels
        raise Exception(f'Found {len(corrected_labels)} instead of 0 or 1')

    if len(corrected_labels) == 0:
        # no previous label exists, so create one
        original_labels = session.query(Label).filter_by(
            image_id=image_id,
            source_id=original_source_id).all()
        assert(len(original_labels) == 1)
        landmark_id = original_labels[0].landmark_id
        label = Label(image_id=image_id,
                      landmark_id=landmark_id,
                      source_id=source.id,
                      x=x, y=y)
    else:
        # one previous label exists, so update it
        assert(len(corrected_labels) == 1)
        label = corrected_labels[0]
        label.x = x
        label.y = y

    session.add(label)
    session.commit()


def get_corrected_source(original_source_id, session):
    sources = session.query(CorrectedLabelSource).filter_by(
        original_source_id=original_source_id).all()

    if len(sources) > 1:
        raise Exception('More than one CorrectedLabelSource found for '
                        f'original_source_id = {original_source_id}')
    elif len(sources) == 1:
        return sources[0]
    else:
        return None


class SmoothedLabelSource(LabelSource):
    __tablename__ = 'smoothed_label_sources'
    id = Column(Integer, ForeignKey('label_sources.id'), primary_key=True)

    source_id = Column(Integer, ForeignKey('label_sources.id'), index=True)
    source = relationship('LabelSource',
                          foreign_keys=[source_id])

    smoother_id = Column(Integer, ForeignKey('smoothers.id'), index=True)
    smoother = relationship('Smoother')

    def __repr__(self):
        return ("<SmoothedLabelSource "
                "id={} name={} source={} smoother={}>".format(
                    self.id, self.name, self.source.name, self.smoother.name))

    @staticmethod
    def find_or_create(session, source_id, smoother_id):
        try:
            smoothed_source = session.query(SmoothedLabelSource)\
                .filter_by(source_id=source_id)\
                .filter_by(smoother_id=smoother_id).one()
        except NoResultFound:
            source = session.query(LabelSource).filter_by(id=source_id).one()
            smoother = session.query(Smoother).filter_by(id=smoother_id).one()
            name = '{}+{}'.format(source.name, smoother.name)
            smoothed_source = SmoothedLabelSource(source_id=source_id,
                                                  smoother_id=smoother_id,
                                                  name=name)
            session.add(smoothed_source)
            session.commit()
        return smoothed_source

    __mapper_args__ = {
        'polymorphic_identity': 'smoothed_labelsource',
        'inherit_condition': (id == LabelSource.id)
    }


class Model(LabelSource):
    __tablename__ = 'models'
    id = Column(Integer, ForeignKey('label_sources.id'), primary_key=True)
    path = Column(String())
    type = Column(String())

    flipped = Column(Boolean)
    test_scale = Column(Float)
    optimize_scale = Column(Boolean)
    optimize_scale_fast = Column(Boolean)
    optimize_scale_image = Column(Boolean)

    image_preproc_type = Column(String())
    training_iters = Column(Integer)
    global_scale = Column(Float)
    augment_bg = Column(Boolean)
    scale_jitter_lo = Column(Float)
    scale_jitter_up = Column(Float)

    trained = Column(Boolean)

    training_set_id = Column(Integer, ForeignKey('imagesets.id'), index=True)
    training_set = relationship('ImageSet')

    labelset_id = Column(Integer, ForeignKey('labelsets.id'), index=True)
    labelset = relationship('LabelSet')

    landmarkset_id = Column(Integer, ForeignKey('landmarksets.id'), index=True)
    landmarkset = relationship('LandmarkSet')

    mirrored = Column(Boolean)

    net = None
    valid_types = ['deepercut']
    image_preproc_types = [
        'none',
        'redify',
        'noise_redify_gray',
        'noise_redify_gray_warp',
        'noise_warp_rotate',
    ]

    __mapper_args__ = {
        'polymorphic_identity': 'model',
    }

    def __init__(self, **kwargs):
        type = kwargs['type']

        if type not in Model.valid_types:
            raise Exception("\'{}\' not recognized.".format(type))

        super().__init__(**kwargs)

    @classmethod
    def display_attrs(self):
        attrs = ['id', 'name', 'training_set_name', 'labelset_name',
                 'training_iters', 'global_scale', 'test_scale',
                 'flipped', 'path']
        return attrs

    @property
    def training_set_name(self):
        if self.training_set_id:
            return self.training_set.name
        else:
            return ''

    @property
    def labelset_name(self):
        if self.labelset_id:
            return self.labelset.name
        else:
            return ''

    def get_net_class(self):
        if self.type == 'deepercut':
            from yogi.nn.models import DeeperCut
            return DeeperCut
        else:
            raise Exception("Model id \'{}\' has invalid type \'{}\'".format(
                self.id, self.type))

    def get_net(self):
        from yogi.config import yogi_dir
        if self.net is None:
            net_class = self.get_net_class()
            path = os.path.join(yogi_dir, self.path)
            self.net = net_class(path=path)
        return self.net

    def get_image_preproc(self):
        from yogi.image_aug import (noise_redify_gray, noise_redify_gray_warp,
                                    redify, noise_warp_rotate)

        image_preproc_dict = {
            None: None,
            'none': None,
            'redify': redify,
            'noise_redify_gray': noise_redify_gray,
            'noise_redify_gray_warp': noise_redify_gray_warp,
            'noise_warp_rotate': noise_warp_rotate
        }

        image_proproc = image_preproc_dict[self.image_preproc_type]

        return image_proproc

    def train(self, labelset, landmarkset, session, cuda_visible_devices=None,
              mirror=False):

        self.get_net().train(labelset, landmarkset,
                             cuda_visible_devices=cuda_visible_devices,
                             augmenter=self.get_image_preproc(),
                             training_iters=self.training_iters,
                             global_scale=self.global_scale,
                             scale_jitter_lo=self.scale_jitter_lo,
                             scale_jitter_up=self.scale_jitter_up,
                             augment_bg=self.augment_bg,
                             mirror=mirror)

        self.trained = True
        self.labelset = labelset
        self.landmarkset = landmarkset

        session.add(self)
        session.commit()

    def apply(self, image, prescale=None, return_scoremap=True):
        from yogi.utils import equals_one

        net = self.get_net()
        if not net.is_loaded:
            net.load()
            assert(net.cfg is not None)

        np_img = image.get_array()

        if self.flipped:
            import numpy as np
            np_img = np.fliplr(np_img)

        # rescale the image prior to testing
        prescale = 1 if prescale is None else prescale
        test_scale = 1 if self.test_scale is None else self.test_scale

        scale = prescale * test_scale

        if not equals_one(scale):
            from scipy.misc import imresize
            np_img = imresize(np_img, scale)

        (scoremap, locref, xs, ys, confidences) = net.apply(
            np_img, return_scoremap=True)

        if self.flipped:
            xs = [1.0 - x for x in xs]
            if return_scoremap:
                scoremap = np.fliplr(scoremap)

        if return_scoremap:
            return (scoremap, locref, xs, ys, confidences)
        else:
            return (xs, ys, confidences)

    def label(self, image, session, return_scoremap=False, scale=None):
        if return_scoremap:
            (scoremap_array, locref, xs, ys, confidences) = self.apply(
                image, return_scoremap=True, prescale=scale)
        else:
            (xs, ys, confidences) = self.apply(
                image, return_scoremap=False, prescale=scale)

        landmark_ids = self.landmarkset.ids(self.mirrored)

        labels = []
        for (x, y, confidence, landmark_id) in zip(xs, ys, confidences,
                                                   landmark_ids):
            label = Label(image_id=image.id, source_id=self.id, x=x, y=y,
                          confidence=confidence, scale=scale,
                          landmark_id=landmark_id)
            labels.append(label)

        if return_scoremap:
            raise NotImplementedError
            # scoremap = ScoreMap.create(
            #     session=session, scoremap_array=scoremap_array,
            #     label_id=label.id)
            # return (label, scoremap)

        return labels

#    def label_imageset(self, imageset, session, save_scoremaps=False,
#                       commit_batch=20):
#
#        unlabeled_images = imageset.get_unlabeled_images(session, self.name)
#        self.label_images(unlabeled_images, session,
#                          save_scoremaps=save_scoremaps,
#                          commit_batch=commit_batch)

    def label_clipset(self, clipset, session, save_scoremaps=False,
                      commit_batch=20):
        for clip in clipset.clips:
            self.label_clip(clip, session, save_scoremaps=save_scoremaps,
                            commit_batch=commit_batch)

    def label_imageset(self, imageset, session, save_scoremaps=False, commit_batch=20):
        from yogi.scaling import optimize_scale, optimize_scale_fast, optimize_scale_image

        print('computing labels for imageset: {}'.format(str(imageset)))

        if self.optimize_scale or self.optimize_scale_fast or self.optimize_scale_image:

            if self.optimize_scale:
                optimizer = optimize_scale

            if self.optimize_scale_fast:
                optimizer = optimize_scale_fast

            if self.optimize_scale_image:
                optimizer = optimize_scale_image

            images = imageset.images
            (best_scale, label_array) = optimizer(images, self)
            self.label_images(images, session, save_scoremaps=save_scoremaps,
                              commit_batch=commit_batch, scale=best_scale,
                              label_array=label_array)
        else:
            # images = clip.ordered_frames
            unlabeled_images = imageset.get_unlabeled_images(session, self.name)
            self.label_images(unlabeled_images, session,
                              save_scoremaps=save_scoremaps,
                              commit_batch=commit_batch)

    def label_clip(self, clip, session, save_scoremaps=False, commit_batch=20):
        from yogi.scaling import optimize_scale, optimize_scale_fast, optimize_scale_image

        if clip.has_labels_from(self):
            print('clip already has labels: {}'.format(str(clip)))
            return

        print('computing labels for clip: {}'.format(str(clip)))

        if self.optimize_scale or self.optimize_scale_fast or self.optimize_scale_image:

            if self.optimize_scale:
                optimizer = optimize_scale

            if self.optimize_scale_fast:
                optimizer = optimize_scale_fast

            if self.optimize_scale_image:
                optimizer = optimize_scale_image

            images = clip.ordered_frames
            (best_scale, label_array) = optimizer(images, self)
            self.label_images(images, session, save_scoremaps=save_scoremaps,
                              commit_batch=commit_batch, scale=best_scale,
                              label_array=label_array)
        else:
            # images = clip.ordered_frames
            unlabeled_images = clip.get_unlabeled_images(session,
                                                         self.name)
            self.label_images(unlabeled_images, session,
                              save_scoremaps=save_scoremaps,
                              commit_batch=commit_batch)

    def label_subclipset(self, subclipset, session, save_scoremaps=False,
                         commit_batch=20):
        for subclip in subclipset.subclips:
            self.label_clip(subclip, session, save_scoremaps=save_scoremaps,
                            commit_batch=commit_batch)
        # from yogi.scaling import optimize_scale, optimize_scale_fast
        #
        # assert(self.optimize_scale or self.optimize_scale_fast)
        #
        # optimizer = (optimize_scale if self.optimize_scale else
        #              optimize_scale_fast)
        #
        # for subclip in subclipset.subclips:
        #     if subclip.has_labels_from(self):
        #         print('subclip already has labels: {}:{}, {}'.format(
        #             subclip.start_idx, subclip.end_idx, subclip.clip.path))
        #         continue
        #     print('computing labels for subclip: {}:{}, {}'.format(
        #         subclip.start_idx, subclip.end_idx, subclip.clip.path))
        #     images = subclip.images
        #     (best_scale, label_array) = optimizer(images, self)
        #     self.label_images(images, session, save_scoremaps=save_scoremaps,
        #                       commit_batch=commit_batch, scale=best_scale,
        #                       label_array=label_array)

    def label_images(self, images, session, save_scoremaps=False,
                     commit_batch=20, label_array=None, scale=None):

        n_unlabeled = len(images)
        print('labeling {} unlabeled images'.format(n_unlabeled))

        all_labels = []
        # scoremaps = []

        for (i, image) in enumerate(images):

            print('labeling image {} / {} ({})'.format(i,
                                                       n_unlabeled,
                                                       image.path))

            if '__len__' in dir(scale):
                image_scale = scale[i]
            else:
                image_scale = scale

            labels = self.label(image, session, return_scoremap=False,
                                scale=image_scale)
            all_labels.extend(labels)

        #for (i, label) in enumerate(all_labels):
        #    session.add(label)
        #    if (i % commit_batch == 0):
        #        session.commit()

        session.bulk_save_objects(all_labels)

        session.commit()

    def __repr__(self):
        return ('<Model id="{}", name="{}", '
                'path="{}", labelset_id="{}", '
                'image_preproc_type="{}", training_iters="{}", '
                'trained="{}">').format(
                    self.id, self.name, self.path, self.labelset_id,
                    self.image_preproc_type, self.training_iters,
                    self.trained)


# class ScoreMap(Base):
#     __tablename__ = 'scoremaps'
#     id = Column(Integer, primary_key=True)
#     label_id = Column(Integer, ForeignKey('labels.id'), index=True)
#
#     height = Column(Integer)
#     width = Column(Integer)
#
#     image_path = Column(String())
#     binary_path = Column(String())
#
#     @staticmethod
#     def create_and_save(session, scoremap_array, label_id):
#         (height, width) = scoremap_array.shape[0:2]
#         image_path = ''
#         binary_path = ''
#         scoremap = ScoreMap(label_id=label_id, height=height, width=width,
#                             image_path=image_path, binary_path=binary_path)
#         session.add(scoremap)
#         session.commit()


class Landmark(Base):
    __tablename__ = 'landmarks'
    id = Column(Integer, primary_key=True)
    name = Column(String())

    mirror_id = Column(Integer, ForeignKey('landmarks.id'))
    mirror = relationship('Landmark', uselist=False)

    label_studio_checkbox_name = Column(String())
    label_studio_label_name = Column(String())

    color = Column(String())


class LandmarkSet(Base):
    __tablename__ = 'landmarksets'
    id = Column(Integer, primary_key=True)
    name = Column(String())
    landmarks = relationship('Landmark',
                             secondary=landmarkset_association_table,
                             order_by=(landmarkset_association_table.c.
                                       landmark_id))

    def is_mirrorable(self):
        """Checks that all the landmarks have a corresponding mirror"""
        ids = [landmark.id for landmark in self.landmarks]
        mirror_ids = [landmark.mirror_id for landmark in self.landmarks]
        return set(ids) == set(mirror_ids)

    def ids(self, mirror):
        if mirror:
            assert(self.is_mirrorable())

        id_list = []
        for landmark in self.landmarks:
            if landmark.id not in id_list:
                id_list.append(landmark.id)
            if mirror:
                if landmark.mirror.id not in id_list:
                    id_list.append(landmark.mirror.id)
        return id_list

    def index(self, landmark_id, mirror):
        if mirror:
            assert(self.is_mirrorable())

        id_list = self.ids(mirror=mirror)
        return id_list.index(landmark_id)

    def all_joints(self, mirror=False):
        if mirror:
            assert(self.is_mirrorable())
        # all_joints: [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12]]
        # get joint groups
        joint_groups = []
        for landmark in self.landmarks:
            idx = self.index(landmark.id, mirror)
            if not any([idx in group for group in joint_groups]):
                group = [idx]
                if mirror:
                    mirror_idx = self.index(landmark.mirror.id, mirror)
                    group.append(mirror_idx)
                joint_groups.append(group)
        return joint_groups

    # def all_joints_names(self, mirror=False):
    #     # all_joints_names: ['ankle', 'knee', 'hip', 'wrist', 'elbow' ...]
    #     pass


class Annotator(Base):
    __tablename__ = 'annotators'
    id = Column(Integer, primary_key=True)
    name = Column(String())


class AnnotationSet(LabelSource):
    __tablename__ = 'annotationsets'
    id = Column(Integer, ForeignKey('label_sources.id'), primary_key=True)

    annotator_id = Column(Integer, ForeignKey('annotators.id'))
    annotator = relationship('Annotator')

    landmark_set_id = Column(Integer, ForeignKey('landmarksets.id'))
    landmark_set = relationship('LandmarkSet')

    __mapper_args__ = {
        'polymorphic_identity': 'annotation_set',
    }

    def add_labels_from_json(self, session, json_files):
        from yogi.label_studio import load_labels

        # load labels from json
        all_labels = []
        for json_file in json_files:
            labels = load_labels(json_file, self.landmark_set.landmarks)
            all_labels.append(labels)

        # add labels to database
        for labels in all_labels:
            for label in labels:
                label.source_id = self.id
                session.add(label)

        session.commit()

    @classmethod
    def display_attrs(self):
        col_names = [col.name for col in self.__table__.columns]
        attrs = ['name'] + col_names
        return attrs


class DyeDetector(LabelSource):
    __tablename__ = 'dye_detectors'
    id = Column(Integer, ForeignKey('label_sources.id'), primary_key=True)
    type = Column(String())

    channel = Column(Integer)
    threshold = Column(Integer)

    erode = Column(Boolean)
    size_threshold = Column(Integer)

    valid_types = ['thresholder']

    __mapper_args__ = {
        'polymorphic_identity': 'dye_detector',
    }

    @classmethod
    def display_attrs(self):
        col_names = [col.name for col in self.__table__.columns]
        attrs = ['name'] + col_names
        return attrs

    def __init__(self, **kwargs):
        type = kwargs['type']

        if type not in DyeDetector.valid_types:
            raise Exception("\'{}\' not recognized.".format(type))

        super().__init__(**kwargs)

    def apply(self, uv_image):
        from yogi.centroids import (get_centroid, image_to_mask, get_area,
                                    get_mean_value)

        mask = image_to_mask(uv_image, threshold=self.threshold,
                             channel=self.channel, erode=self.erode,
                             size_threshold=self.size_threshold)
        (x, y) = get_centroid(mask)
        area = get_area(mask)
        mean_value = get_mean_value(uv_image, self.channel, mask)
        return (x, y, area, mean_value)

    def __repr__(self):
        return ('<DyeDetector id="{}", name="{}", '
                'type="{}", channel="{}", threshold="{}">').format(
            self.id, self.name, self.type, self.channel, self.threshold)


class Plot(Base):
    __tablename__ = 'plots'
    id = Column(Integer, primary_key=True)
    path = Column(String())
    plot_type = Column(String(50))

    valid_types = ['roc-curve']

    __mapper_args__ = {
        'polymorphic_identity': 'plot',
        'polymorphic_on': plot_type,
    }


class RocCurve(Plot):
    __tablename__ = 'roc_curves'
    id = Column(Integer, ForeignKey('plots.id'), primary_key=True)
    imageset_id = Column(Integer, ForeignKey('imagesets.id'))
    source_id_pred = Column(Integer, ForeignKey('label_sources.id'))
    source_id_gt = Column(Integer, ForeignKey('label_sources.id'))

    imageset = relationship("ImageSet")
    source_pred = relationship("LabelSource", foreign_keys=[source_id_pred])
    source_gt = relationship("LabelSource", foreign_keys=[source_id_gt])

    __mapper_args__ = {
        'polymorphic_identity': 'roc-curve',
    }

    def __repr__(self):
        return ('<RocCurve id="{}", path="{}">').format(self.id, self.path)

    def generate(self, **kwargs):
        from matplotlib import pyplot as plt
        from yogi.evaluation import plot_roc

        fig = plt.figure()

        plot_roc(self.imageset.name, self.source_pred.name,
                 self.source_gt.name, **kwargs)

        path_dir = os.path.dirname(self.path)
        os.makedirs(path_dir, exist_ok=True)

        plt.savefig(self.path)
        plt.close(fig)


class BoundingBox(Base):

    __tablename__ = 'bounding_boxes'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), index=True)

    annotator_id = Column(Integer, ForeignKey('annotators.id'))
    annotator = relationship('Annotator')

    x = Column(Integer)
    y = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)

    @staticmethod
    def create_from_json(session, json_files, annotator_id):
        from yogi.label_studio import load_bounding_box

        # load bounding box from json
        for json_file in json_files:
            print('loading {}'.format(json_file))
            bounding_box = load_bounding_box(json_file)
            bounding_box.annotator_id = annotator_id
            session.add(bounding_box)

        session.commit()

