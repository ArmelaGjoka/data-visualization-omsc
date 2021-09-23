########################### DO NOT MODIFY THIS SECTION ##########################
#################################################################################
import sqlite3
from sqlite3 import Error
import csv
#################################################################################

## Change to False to disable Sample
SHOW = True

############### SAMPLE CLASS AND SQL QUERY ###########################
######################################################################
class Sample():
    def sample(self):
        try:
            connection = sqlite3.connect("sample")
            connection.text_factory = str
        except Error as e:
            print("Error occurred: " + str(e))
        print('\033[32m' + "Sample: " + '\033[m')
        
        # Sample Drop table
        connection.execute("DROP TABLE IF EXISTS sample;")
        # Sample Create
        connection.execute("CREATE TABLE sample(id integer, name text);")
        # Sample Insert
        connection.execute("INSERT INTO sample VALUES (?,?)",("1","test_name"))
        connection.commit()
        # Sample Select
        cursor = connection.execute("SELECT * FROM sample;")
        print(cursor.fetchall())

######################################################################

class HW2_sql():
    ############### DO NOT MODIFY THIS SECTION ###########################
    ######################################################################
    def create_connection(self, path):
        connection = None
        try:
            connection = sqlite3.connect(path)
            connection.text_factory = str
        except Error as e:
            print("Error occurred: " + str(e))
    
        return connection

    def execute_query(self, connection, query):
        cursor = connection.cursor()
        try:
            print(query)
            if query == "":
                return "Query Blank"
            else:
                cursor.execute(query)
                connection.commit()
                return "Query executed successfully"
        except Error as e:
            return "Error occurred: " + str(e)
    ######################################################################
    ######################################################################

    # GTusername [0 points]
    def GTusername(self):
        gt_username = "agjoka3"
        return gt_username
    
    # Part a.i Create Tables [2 points]
    def part_ai_1(self,connection):
        part_ai_1_sql = "CREATE TABLE movies (id INTEGER, title TEXT, score REAL)"
        
        return self.execute_query(connection, part_ai_1_sql)

    def part_ai_2(self,connection):
        part_ai_2_sql = "CREATE TABLE movie_cast (movie_id INTEGER, cast_id INTEGER, cast_name TEXT, birthday TEXT, popularity REAL)"
        
        return self.execute_query(connection, part_ai_2_sql)
    
    # Part a.ii Import Data Movies [2 points]
    def part_aii_1(self,connection,path):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                connection.execute("INSERT INTO movies VALUES (?,?,?)",(row[0], row[1], row[2]))
                connection.commit()
        
        sql = "SELECT COUNT(id) FROM movies;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]
    
    def part_aii_2(self,connection, path):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                connection.execute("INSERT INTO movie_cast VALUES (?,?,?,?,?)",(row[0], row[1], row[2], row[3], row[4]))
                connection.commit()

        sql = "SELECT COUNT(cast_id) FROM movie_cast;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]

    # Part a.iii Vertical Database Partitioning [5 points]
    def part_aiii(self,connection):

        part_aiii_sql = """CREATE TABLE cast_bio 
                (cast_id int PRIMARY KEY, cast_name TEXT, birthday TEXT, popularity REAL)"""       
        self.execute_query(connection, part_aiii_sql)
        
        ############### CREATE IMPORT CODE BELOW ############################
        part_aiii_insert_sql = """INSERT INTO cast_bio(cast_id, cast_name, birthday, popularity) 
                                  SELECT DISTINCT(cast_id), cast_name, birthday, popularity from  movie_cast"""
        ######################################################################
        
        self.execute_query(connection, part_aiii_insert_sql)
        
        sql = "SELECT COUNT(cast_id) FROM cast_bio;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]
       

    # Part b Create Indexes [1 points]
    def part_b_1(self,connection):
        part_b_1_sql = "CREATE INDEX movie_index ON movies (id)"
        return self.execute_query(connection, part_b_1_sql)
    
    def part_b_2(self,connection):
        part_b_2_sql = "CREATE INDEX cast_index ON movie_cast (cast_id)"
        return self.execute_query(connection, part_b_2_sql)
    
    def part_b_3(self,connection):
        part_b_3_sql = "CREATE INDEX cast_bio_index ON cast_bio (cast_id)"
        return self.execute_query(connection, part_b_3_sql)
    
    # Part c Calculate a Proportion [3 points]
    def part_c(self,connection):
        part_test_sql = """WITH val as 
                                (SELECT AVG(CASE WHEN score > 50 and title LIKE '%war%' THEN 1.00 ELSE 0.00 end) *100 as ratio from movies)
                                SELECT printf('%.2f', ratio) FROM val"""
        cursor = connection.execute(part_test_sql)
        return cursor.fetchall()[0][0]

    # Part d Find the Most Prolific Actors [4 points]
    def part_d(self,connection):
        part_d_sql = """ SELECt cast_name, COUNT(movie_id) as cast_movies
                            FROM movie_cast
                            GROUp BY cast_id
                            HAVING popularity > 10
                            ORDER BY cast_movies DESC, cast_name ASC
                            LIMIT 5
        """
        cursor = connection.execute(part_d_sql)
        return cursor.fetchall()

    # Part e Find the Highest Scoring Movies With the Least Amount of Cast [4 points]
    def part_e(self,connection):
        part_e_sql = """ 
                    SELECT movies.title, printf('%.2f', movies.score), COUNT(movie_cast.cast_name) as cast_count
                    FROM movies 
                    JOIN movie_cast
                    ON movies.id = movie_cast.movie_id
                    GROUP By movies.id
                    ORDER BY movies.score DESC, cast_count ASC, movies.title ASC
                    LIMIT 5
                    """
        cursor = connection.execute(part_e_sql)
        return cursor.fetchall()
    
    # Part f Get High Scoring Actors [4 points]
    def part_f(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_f_sql = """
                        SELECT movie_cast.cast_id, movie_cast.cast_name, printf('%.2f',AVG(movies.score))
                        FROM movie_cast
                        JOIN movies ON movie_cast.movie_id = movies.id
                        WHERE movies.score >= 25
                        GROUP BY movie_cast.cast_id
                        HAVING COUNT(movies.id) > 2
                        ORDER BY AVG(movies.score) DESC, movie_cast.cast_name ASC
                        LIMIT 10
                    """
        ######################################################################
        cursor = connection.execute(part_f_sql)
        return cursor.fetchall()

    # Part g Creating Views [6 points]
    def part_g(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_g_sql = """ 
            CREATE VIEW good_collaboration (cast_member_id1, cast_member_id2, movie_count, average_movie_score) AS
            SELECT cast_member_id1, cast_member_id2, COUNT(movie_id) as movie_count,printf('%.2f', AVG(movies.score)) as average_movie_score
            FROM
            (SELECT CAST(c1.cast_id AS INTEGER) as cast_member_id1, CAST(c2.cast_id AS INTEGER) as cast_member_id2, c1.movie_id as movie_id
            FROM movie_cast c1
            INNER JOIN movie_cast c2 ON c1.movie_id = c2.movie_id AND cast_member_id1 < cast_member_id2) AS ROWS
            JOIN movies ON movies.id = movie_id
            GROUP BY cast_member_id1, cast_member_id2
            HAVING AVG(movies.score) >= 40 AND COUNT(movies.id) >= 3
        """
        ######################################################################
        return self.execute_query(connection, part_g_sql)
    
    def part_gi(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_g_i_sql = """
        			SELECT CAST(c_id AS INTEGER) as cast_id, movie_cast.cast_name as cast_name, printf('%.2f', score) AS collaboration_score  
                    FROM (
                    SELECT * FROM 
                	(SELECT cast_member_id1 AS c_id, AVG(average_movie_score) AS score
               		FROM good_collaboration GROUP BY c_id 
                	UNION 
                	SELECT cast_member_id2 AS c_id, AVG(average_movie_score) AS score
               		FROM good_collaboration GROUP BY c_id))
                    JOIN movie_cast
                    ON movie_cast.cast_id = c_id
                    GROUP BY cast_id
                    ORDER BY collaboration_score DESC, cast_name ASC
                    LIMIT 5
        """
        ######################################################################
        cursor = connection.execute(part_g_i_sql)
        return cursor.fetchall()
    
    # Part h FTS [4 points]
    def part_h(self,connection,path):
        part_h_sql = "CREATE VIRTUAL TABLE movie_overview USING fts3(id, overview);"
        connection.execute(part_h_sql)
        
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                connection.execute("INSERT INTO movie_overview VALUES (?,?)",(row[0], row[1]))
                connection.commit()

        sql = "SELECT COUNT(id) FROM movie_overview;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]
        
    def part_hi(self,connection):
        part_hi_sql = "SELECT COUNT(id) FROM movie_overview WHERE overview MATCH 'fight';"
        cursor = connection.execute(part_hi_sql)
        return cursor.fetchall()[0][0]
    
    def part_hii(self,connection):
        part_hii_sql = "SELECT COUNT(id) FROM movie_overview WHERE overview match 'space NEAR/5 program';"
        cursor = connection.execute(part_hii_sql)
        return cursor.fetchall()[0][0]


if __name__ == "__main__":
    
    ########################### DO NOT MODIFY THIS SECTION ##########################
    #################################################################################
    if SHOW == True:
        sample = Sample()
        sample.sample()

    print('\033[32m' + "Q2 Output: " + '\033[m')
    db = HW2_sql()
    try:
        conn = db.create_connection("Q2")
    except:
        print("Database Creation Error")

    try:
        conn.execute("DROP TABLE IF EXISTS movies;")
        conn.execute("DROP TABLE IF EXISTS movie_cast;")
        conn.execute("DROP TABLE IF EXISTS cast_bio;")
        conn.execute("DROP VIEW IF EXISTS good_collaboration;")
        conn.execute("DROP TABLE IF EXISTS movie_overview;")
    except:
        print("Error in Table Drops")

    try:
        print('\033[32m' + "part ai 1: " + '\033[m' + str(db.part_ai_1(conn)))
        print('\033[32m' + "part ai 2: " + '\033[m' + str(db.part_ai_2(conn)))
    except:
         print("Error in Part a.i")

    try:
        print('\033[32m' + "Row count for Movies Table: " + '\033[m' + str(db.part_aii_1(conn,"data/movies.csv")))
        print('\033[32m' + "Row count for Movie Cast Table: " + '\033[m' + str(db.part_aii_2(conn,"data/movie_cast.csv")))
    except:
        print("Error in part a.ii")

    try:
        print('\033[32m' + "Row count for Cast Bio Table: " + '\033[m' + str(db.part_aiii(conn)))
    except:
        print("Error in part a.iii")

    try:
        print('\033[32m' + "part b 1: " + '\033[m' + db.part_b_1(conn))
        print('\033[32m' + "part b 2: " + '\033[m' + db.part_b_2(conn))
        print('\033[32m' + "part b 3: " + '\033[m' + db.part_b_3(conn))
    except:
        print("Error in part b")

    try:
        print('\033[32m' + "part c: " + '\033[m' + str(db.part_c(conn)))
    except:
        print("Error in part c")

    try:
        print('\033[32m' + "part d: " + '\033[m')
        for line in db.part_d(conn):
            print(line[0],line[1])
    except:
        print("Error in part d")

    try:
        print('\033[32m' + "part e: " + '\033[m')
        for line in db.part_e(conn):
            print(line[0],line[1],line[2])
    except:
        print("Error in part e")

    try:
        print('\033[32m' + "part f: " + '\033[m')
        for line in db.part_f(conn):
            print(line[0],line[1],line[2])
    except:
        print("Error in part f")
    
    try:
        #print('\033[32m' + "part g: " + '\033[m' + str(db.part_g(conn)))
        #print('\033[32m' + "part g.i: " + '\033[m')
        for line in db.part_gi(conn):
            print(line[0],line[1],line[2])
    except:
        print("Error in part g")

    try:   
        print('\033[32m' + "part h.i: " + '\033[m'+ str(db.part_h(conn,"data/movie_overview.csv")))
        print('\033[32m' + "Count h.ii: " + '\033[m' + str(db.part_hi(conn)))
        print('\033[32m' + "Count h.iii: " + '\033[m' + str(db.part_hii(conn)))
    except:
        print("Error in part h")

    conn.close()
    #################################################################################
    #################################################################################
  
