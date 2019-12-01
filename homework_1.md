# 1

![image](https://github.com/Phyllislai12/1801212867-PHBS_BIGDATA_2019/blob/master/photo/hw1-1.png)
![image](https://github.com/Phyllislai12/1801212867-PHBS_BIGDATA_2019/blob/master/photo/hw1-2.png)
![image](https://github.com/Phyllislai12/1801212867-PHBS_BIGDATA_2019/blob/master/photo/hw1-3.png)

# 2
![image](https://github.com/Phyllislai12/1801212867-PHBS_BIGDATA_2019/blob/master/photo/hw1-4.png)

# 1
How many airlines are there（UniqueCarrier）？
select UniqueCarrier, count（*）（*） from statistics group by UniqueCarrier;
![image](https://github.com/Phyllislai12/1801212867-PHBS_BIGDATA_2019/blob/master/photo/hw1-5.png)

# 2
Which airline has the longest flight distance? What is the distance?
select UniqueCarrier,sum(Distance) from statistics group by UniqueCarrier limit 100;
![image](https://github.com/Phyllislai12/1801212867-PHBS_BIGDATA_2019/blob/master/photo/hw1-6.png)

# 3
How many carriers in all single records have a departure delay greater than the average of all records? code can only return one number
hive> set hive.mapred.mode=nonstrict;
hive> select count(temp.avg_dep) from statistics t1 left join (select avg(t2.DepTime) avg_dep from statistics t2) temp on t1.DepTime>temp.avg_dep;
![image](https://github.com/Phyllislai12/1801212867-PHBS_BIGDATA_2019/blob/master/photo/hw1-7.png)

# 4
Please find the total number of cancellations per airline, per destination. Show only the 5 worst records
 select * from (
                  select UniqueCarrier,
                         Origin,
                         sum(CarrierDelay + WeatherDelay + NASDelay + SecurityDelay + LateAircraftDelay) delay_count
                  from statistics
                  group by UniqueCarrier, Origin
              ) r order by delay_count desc limit 5;
![image](https://github.com/Phyllislai12/1801212867-PHBS_BIGDATA_2019/blob/master/photo/hw1-8.png)
