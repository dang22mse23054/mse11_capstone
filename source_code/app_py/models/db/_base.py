import os
import traceback
from datetime import datetime, timezone
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import insert

ENV = dotenv_values('../app/.env')
TIME_ZONE = ENV["TIME_ZONE"]
DB_USER = ENV['DB_USERNAME']
DB_PASSWORD = ENV['DB_PASSWORD']
DB_NAME = ENV['DB_NAME']

DB_HOST = ENV['DB_HOST']
DB_PORT = ENV['DB_PORT']

DB_RO_HOST = ENV['DB_HOST']
DB_RO_PORT = ENV['DB_PORT']
BULK_MAX = 1000

# Config echo=True in create_engine to print DB execution query INFO

# Master
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4",convert_unicode=True, echo=True)
Session = sessionmaker(bind=engine, expire_on_commit=False)

# Slave
readonly_engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_RO_HOST}:{DB_RO_PORT}/{DB_NAME}?charset=utf8mb4",convert_unicode=True, echo=True)
ROSession = sessionmaker(bind=readonly_engine, expire_on_commit=False)

Base = declarative_base()
class DbBase(Base):
    __abstract__ = True

    @classmethod
    def get_session(cls, readonly = False):
        return ROSession() if readonly else Session()

    @classmethod
    def all(cls, model):
        session = DbBase.get_session(True)
        try:
            return session.query(model).all()
        finally:
            session.close()

    @classmethod
    def where(cls, model, condition, limit=None):
        session = DbBase.get_session(True)
        try:
            if limit:
                return session.query(model).filter_by(**condition).limit(limit)
            return session.query(model).filter_by(**condition).all()
        finally:
            session.close()
    
    @classmethod
    def filter(cls, model, condition, limit=None):
        session = DbBase.get_session(True)
        try:
            if limit:
                return session.query(model).filter(*condition).limit(limit)
            return session.query(model).filter(*condition).all()
        finally:
            session.close()

    @classmethod
    def soft_delete(cls, model, condition, sess=None):
        session = DbBase.get_session() if sess == None else sess
        try:
            session.query(model) \
                .filter(*condition) \
                .update({"deletedAt": datetime.now(timezone.utc)}, synchronize_session=False)
            session.commit()
            return True
        except Exception as ex:
            session.rollback()
            traceback.print_exc()
            return False

    @classmethod
    def update(cls, model, condition, update_values, sess = None):
        session = DbBase.get_session()
        record = session.query(model).filter_by(**condition).scalar()
        try:
            if not record:
                raise Exception("Record not found!")
            session.query(model).filter_by(id=record.id).update(update_values)
            session.commit()
            return True
        except Exception as ex:
            session.rollback()
            print("Failed to update record due to: %s" % ex)
            traceback.print_exc()
            return False
    
    @classmethod
    def bulk_insert(cls, model, columns = [], data = [], sess = None):
        idx = 0
        record_len = len(data)
        session = DbBase.get_session() if sess == None else sess

        try:
            while idx < record_len:
                # session.insert(record)
                last_record = idx + BULK_MAX if record_len > idx + BULK_MAX else record_len
                insert_record = [dict(zip(columns, d)) for d in data[idx:last_record]]
                insert_stmt = insert(model).values(insert_record)
                session.execute(insert_stmt)
                idx = idx + BULK_MAX
            session.commit()
        except Exception as ex:
            session.rollback()
            print("Failed to create record due to: %s" % ex)
            traceback.print_exc()
            return False
        return True

    @classmethod
    def exec_custom_query(cls, query, readonly = False):
        session = DbBase.get_session(readonly)
        try:
            data = session.execute(query)
            session.commit()
            return data
        except Exception as ex:
            session.rollback()
            print("Fail to execute query: %s" % query)
            traceback.print_exc()
            return []
        finally:
            session.close()
