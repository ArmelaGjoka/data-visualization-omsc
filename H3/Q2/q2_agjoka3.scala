// Databricks notebook source
// STARTER CODE - DO NOT EDIT THIS CELL
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import spark.implicits._
import org.apache.spark.sql.expressions.Window

// COMMAND ----------

// STARTER CODE - DO NOT EDIT THIS CELL
val customSchema = StructType(Array(StructField("lpep_pickup_datetime", StringType, true), StructField("lpep_dropoff_datetime", StringType, true), StructField("PULocationID", IntegerType, true), StructField("DOLocationID", IntegerType, true), StructField("passenger_count", IntegerType, true), StructField("trip_distance", FloatType, true), StructField("fare_amount", FloatType, true), StructField("payment_type", IntegerType, true)))

// COMMAND ----------

// STARTER CODE - YOU CAN LOAD ANY FILE WITH A SIMILAR SYNTAX.
val df = spark.read
   .format("com.databricks.spark.csv")
   .option("header", "true") // Use first line of all files as header
   .option("nullValue", "null")
   .schema(customSchema)
   .load("/FileStore/tables/nyc_tripdata.csv") // the csv file which you want to work with
   .withColumn("pickup_datetime", from_unixtime(unix_timestamp(col("lpep_pickup_datetime"), "MM/dd/yyyy HH:mm")))
   .withColumn("dropoff_datetime", from_unixtime(unix_timestamp(col("lpep_dropoff_datetime"), "MM/dd/yyyy HH:mm")))
   .drop($"lpep_pickup_datetime")
   .drop($"lpep_dropoff_datetime")

// COMMAND ----------

// LOAD THE "taxi_zone_lookup.csv" FILE SIMILARLY AS ABOVE. CAST ANY COLUMN TO APPROPRIATE DATA TYPE IF NECESSARY.

// ENTER THE CODE BELOW
val taxicustomSchema = StructType(Array(StructField("LocationID", StringType, true), StructField("Borough", StringType, true), StructField("Zone", StringType, true), StructField("service_zone", StringType, true)))
val taxidf = spark.read
   .format("com.databricks.spark.csv")
   .option("header", "true") // Use first line of all files as header
   .option("nullValue", "null")
   .schema(taxicustomSchema)
   .load("/FileStore/tables/taxi_zone_lookup.csv")
taxidf.show(5)

// COMMAND ----------

// STARTER CODE - DO NOT EDIT THIS CELL
// Some commands that you can use to see your dataframes and results of the operations. You can comment the df.show(5) and uncomment display(df) to see the data differently. You will find these two functions useful in reporting your results.
// display(df)
df.show(5) // view the first 5 rows of the dataframe

// COMMAND ----------

// STARTER CODE - DO NOT EDIT THIS CELL
// Filter the data to only keep the rows where "PULocationID" and the "DOLocationID" are different and the "trip_distance" is strictly greater than 2.0 (>2.0).

// VERY VERY IMPORTANT: ALL THE SUBSEQUENT OPERATIONS MUST BE PERFORMED ON THIS FILTERED DATA

val df_filter = df.filter($"PULocationID" =!= $"DOLocationID" && $"trip_distance" > 2.0)
df_filter.show(5)

// COMMAND ----------

// PART 1a: The top-5 most popular drop locations - "DOLocationID", sorted in descending order - if there is a tie, then one with lower "DOLocationID" gets listed first
// Output Schema: DOLocationID int, number_of_dropoffs int 

// Hint: Checkout the groupBy(), orderBy() and count() functions.

// ENTER THE CODE BELOW
val df_top_do = df_filter.groupBy("DOLocationID").count().alias("Count").orderBy(desc("Count"), desc("DOLocationID")).limit(5)
df_top_do.show()

// COMMAND ----------

// PART 1b: The top-5 most popular pickup locations - "PULocationID", sorted in descending order - if there is a tie, then one with lower "PULocationID" gets listed first 
// Output Schema: PULocationID int, number_of_pickups int

// Hint: Code is very similar to part 1a above.

// ENTER THE CODE BELOW
val df_top_pu = df_filter.groupBy("PULocationID").count().alias("Count").orderBy(desc("Count"), desc("PULocationID")).limit(5)
df_top_pu.show()

// COMMAND ----------

// PART 2: List the top-3 locations with the maximum overall activity, i.e. sum of all pickups and all dropoffs at that LocationID. In case of a tie, the lower LocationID gets listed first.
// Output Schema: LocationID int, number_activities int

// Hint: In order to get the result, you may need to perform a join operation between the two dataframes that you created in earlier parts (to come up with the sum of the number of pickups and dropoffs on each location). 

// ENTER THE CODE BELOW
val all = df_filter.join(taxidf,df_filter("PULocationID") ===  taxidf("LocationID") || df_filter("DOLocationID") === taxidf("LocationID"),"inner").groupBy("LocationID")
.agg(count(lit(1)).alias("number_activities")).orderBy(desc("number_activities"), asc("LocationID")).limit(3)
all.show()

// COMMAND ----------

// PART 3: List all the boroughs in the order of having the highest to lowest number of activities (i.e. sum of all pickups and all dropoffs at that LocationID), along with the total number of activity counts for each borough in NYC during that entire period of time.
// Output Schema: Borough string, total_number_activities int

// Hint: You can use the dataframe obtained from the previous part, and will need to do the join with the 'taxi_zone_lookup' dataframe. Also, checkout the "agg" function applied to a grouped dataframe.

// ENTER THE CODE BELOW
val allBoroughs = df_filter.join(taxidf,df_filter("PULocationID") ===  taxidf("LocationID") || df_filter("DOLocationID") === taxidf("LocationID"),"inner").groupBy("Borough")
.agg(count(lit(1)).alias("number_activities")).orderBy(desc("number_activities"))
allBoroughs.show()

// COMMAND ----------

// PART 4: List the top 2 days of week with the largest number of (daily) average pickups, along with the values of average number of pickups on each of the two days. The day of week should be a string with its full name, for example, "Monday" - not a number 1 or "Mon" instead.
// Output Schema: day_of_week string, avg_count float

// Hint: You may need to group by the "date" (without time stamp - time in the day) first. Checkout "to_date" function.

// ENTER THE CODE BELOW
val allDays = df_filter.select(to_date(col("pickup_datetime")).alias("toDate")).groupBy("toDate").agg(count(lit(1)).alias("dateCount"))
val topTwoDays = allDays.groupBy(date_format(col("toDate"), "EEEE").alias("day_of_week")).agg(avg("dateCount").alias("avg_count")).orderBy(desc("avg_count")).limit(2)
topTwoDays.show()

// COMMAND ----------

// PART 5: For each particular hour of a day (0 to 23, 0 being midnight) - in their order from 0 to 23, find the zone in Brooklyn borough with the LARGEST number of pickups. 
// Output Schema: hour_of_day int, zone string, max_count int

// Hint: You may need to use "Window" over hour of day, along with "group by" to find the MAXIMUM count of pickups

// ENTER THE CODE BELOW
val brooklynBoroughs = df_filter.join(taxidf,df_filter("PULocationID") ===  taxidf("LocationID"),"inner").filter(col("Borough") === "Brooklyn").select(col("Zone"), hour(col("pickup_datetime")).alias("hour")).groupBy("hour", "Zone").agg(count(lit(1)).alias("count"))
val maxDf = brooklynBoroughs.groupBy(col("hour").alias("mHour")).agg(max(col("count")).alias("max")).orderBy(desc("max"))
val result = maxDf.join(brooklynBoroughs, maxDf("mHour") === brooklynBoroughs("hour") && maxDf("max") === brooklynBoroughs("count")).select("hour", "Zone", "max").orderBy(asc("hour"))

result.show(24)

// COMMAND ----------

// PART 6 - Find which 3 different days of the January, in Manhattan, saw the largest percentage increment in pickups compared to previous day, in the order from largest increment % to smallest increment %. 
// Print the day of month along with the percent CHANGE (can be negative), rounded to 2 decimal places, in number of pickups compared to previous day.
// Output Schema: day int, percent_change float
val mb = df_filter.join(taxidf,df_filter("PULocationID") ===  taxidf("LocationID"),"inner").filter(col("Borough") === "Manhattan" && month(col("pickup_datetime")) === 1)
.select(dayofmonth(col("pickup_datetime")).alias("day")).groupBy("day").agg(count(lit(1)).alias("count")).orderBy("day")
val mmb = mb.withColumn("pCount", lag(mb("count"), 1).over(Window.orderBy(mb("day"))))
val mbb = mmb.select(col("day"), ((col("count")/col("pCount") - 1) * 100).alias("percentage_change")).orderBy(desc("percentage_change")).limit(3)
mbb.show()
// Hint: You might need to use lag function, over a window ordered by day of month.

// ENTER THE CODE BELOW
