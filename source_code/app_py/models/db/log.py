import sys
sys.path.append('../../')

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, UniqueConstraint
from utils.timer import TimeUtils
from ._base import DbBase

class Log(DbBase):
    __tablename__ = "log"
    
    id = Column(String(100), primary_key=True, nullable=False)
    videoId = Column(Integer, nullable=False)
    gender = Column(Text, nullable=False)
    age = Column(Text, nullable=False)
    happy = Column(Text, nullable=False)
    createdAt = Column(DateTime, nullable=False)

    def __init__(self, id, videoId, gender, age, happy, createdAt = TimeUtils.now('UTC')):
        self.id = id
        self.videoId = videoId
        self.gender = gender
        self.age = age
        self.happy = happy
        self.createdAt = createdAt
    
    def __repr(self):
        return "<Log(id=%s, videoId=%d, gender=%s, age=%s, happy=%s, createdAt=%s)>" % (
                self.id, self.videoId, self.gender, self.age, self.happy, self.createdAt)