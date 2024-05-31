import sys
sys.path.append('../../')

from sqlalchemy import create_engine, Column, Integer, Text, DateTime, PrimaryKeyConstraint
from utils.timer import TimeUtils
from ._base import DbBase

class Statistic(DbBase):
    __tablename__ = "statistic"
    
    videoId = Column(Integer, nullable=False)
    group = Column(Text, nullable=False)
    gender = Column(Text, nullable=False)
    age = Column(Text, nullable=False)
    happy = Column(Text, nullable=False)
    createdAt = Column(DateTime, nullable=False)
    deletedAt = Column(DateTime, nullable=True)
    
    PrimaryKeyConstraint(videoId, group, createdAt, name="video_group_createdAt_pk")

    def __init__(self, videoId, group, gender, age, happy, createdAt = TimeUtils.now('UTC'), deletedAt = None):
        self.videoId = videoId
        self.group = group
        self.gender = gender
        self.age = age
        self.happy = happy
        self.createdAt = createdAt
        self.deletedAt = deletedAt
    
    def __repr(self):
        return "<Statistic(videoId=%d, group=%s, gender=%s, age=%s, happy=%s, createdAt=%s, deletedAt=%s)>" % (
                self.videoId, self.group, self.gender, self.age, self.happy, self.createdAt, self.deletedAt)