import sqlite3


def insert_comment(goods_id, comment, user_id, db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    sql = 'insert into comment (goods_id, comment, user_id)' \
          'values(?,?,?)'
    data = (goods_id, comment, user_id)
    cur.execute(sql, data)
    conn.commit()
    cur.close()
    conn.close()


def return_all(db_file):
    conn = sqlite3.connect(db_file)
    sql = 'select * from comment'
    cur = conn.cursor()
    cur.execute(sql)
    l = cur.fetchall()
    cur.close()
    conn.close()
    return l


def search_goods_comment(goods_id, db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    sql = 'select * from comment where goods_id=?'
    comment = (goods_id,)
    cur.execute(sql, comment)
    l = []
    for item in cur.fetchall():
        l.append(item)
    cur.close()
    conn.close()
    return l


def delete_comment(db_file, goods_id, user_id):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    sql = 'delete from comment where user_id=?&goods_id=?'
    comment = (user_id, goods_id)
    cur.execute(sql, comment)
    conn.commit()
    cur.close()
    conn.close()

