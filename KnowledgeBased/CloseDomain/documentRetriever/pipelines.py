from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from importlib import reload

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pymysql
from pymysql import connections
import settings

# connect the document retriever to database
class DocumentRetrieverPipeline(object):
    def __init__(self) -> None:
        self.conn = pymysql.connect(
            host=settings.HOST_IP,
            port=settings.PORT,
            user=settings.USER,
            passwd=settings.PASSWD,
            db=settings.DB_NAME,
            charset='utf8mb4',
            use_unicode=True
        )
        self.cursor = self.conn.cursor()

    def process_item(self, item, spider):
        pass