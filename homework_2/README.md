To run the following code, `pip install pandas, numpy, seaborn, duckdb, matplotlib, and cassandra-sigv4`. The expected outputs are as follow


```python
!pip install duckdb==1.2.2
import duckdb
```

### Part I


```python
csv_folder = r"C:\Users\siddh\OneDrive - Northwestern University\Desktop\DATA ENG 300\HW2\mimic-iii-clinical-database-demo-1.4\mimic-iii-clinical-database-demo-1.4"

conn = duckdb.connect("mimiciii.db")
files = [
    "ADMISSIONS.csv",
    "ICUSTAYS.csv",
    "PATIENTS.csv",
    "PROCEDURES_ICD.csv",
    "D_ICD_PROCEDURES.csv",
    "DRGCODES.csv",
    "PRESCRIPTIONS.csv"
]

for file in files:
    table_name = file.replace(".csv", "").lower()
    file_path = f"{csv_folder}\\{file}"  
    conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_csv_auto('{file_path}')
    """)
```

#### Question 1


```python
conn.sql(
    """
    SELECT DISTINCT ethnicity
    FROM admissions
    ORDER BY ethnicity;
    """
)

```




    ┌──────────────────────────────────────────────────────────┐
    │                        ethnicity                         │
    │                         varchar                          │
    ├──────────────────────────────────────────────────────────┤
    │ AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE │
    │ ASIAN                                                    │
    │ BLACK/AFRICAN AMERICAN                                   │
    │ HISPANIC OR LATINO                                       │
    │ HISPANIC/LATINO - PUERTO RICAN                           │
    │ OTHER                                                    │
    │ UNABLE TO OBTAIN                                         │
    │ UNKNOWN/NOT SPECIFIED                                    │
    │ WHITE                                                    │
    └──────────────────────────────────────────────────────────┘




```python
conn.sql(
    """
    CREATE OR REPLACE TABLE admissions_cleaned AS
    SELECT *,
        CASE
            WHEN LOWER(ethnicity) IN (
                'unknown', 
                'unknown/not specified', 
                'unable to obtain', 
                'other', 
                'other/unknown'
            ) THEN 'Other/Unknown'
            ELSE ethnicity
        END AS ethnicity_cleaned
    FROM admissions;
    """
)

```


```python
conn.sql(
    """
    CREATE OR REPLACE TABLE prescriptions_cleaned AS
    SELECT 
        pr.hadm_id,
        pr.drug,
        TRY_CAST(pr.dose_val_rx AS DOUBLE) AS dose,
        ac.ethnicity_cleaned AS ethnicity
    FROM prescriptions pr
    JOIN admissions_cleaned ac
        ON pr.hadm_id = ac.hadm_id
    WHERE pr.dose_val_rx IS NOT NULL;
    """
)

```


```python
conn.sql(
    """
    CREATE OR REPLACE TABLE drug_totals_by_ethnicity AS
    SELECT
        ethnicity,
        drug,
        SUM(dose) AS total_amount
    FROM prescriptions_cleaned
    GROUP BY ethnicity, drug;
    """
)

```


```python
amount_by_ethnicity_df = conn.sql(
    """
    SELECT *
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY ethnicity ORDER BY total_amount DESC) AS rn
        FROM drug_totals_by_ethnicity
    )
    WHERE rn = 1
    ORDER BY ethnicity;
    """
).df()
amount_by_ethnicity_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ethnicity</th>
      <th>drug</th>
      <th>total_amount</th>
      <th>rn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGN...</td>
      <td>5% Dextrose</td>
      <td>16900.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ASIAN</td>
      <td>Heparin</td>
      <td>15000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BLACK/AFRICAN AMERICAN</td>
      <td>Heparin Sodium</td>
      <td>150000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HISPANIC OR LATINO</td>
      <td>5% Dextrose</td>
      <td>19950.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HISPANIC/LATINO - PUERTO RICAN</td>
      <td>0.9% Sodium Chloride</td>
      <td>43663.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



 (b) I have written a series of smaller queries to answer question 1. To find the total drugs and their amount used by ethnicity, the query first groups together the `"UNKNOWN", "UNKNOWN/NOT SPECIFIED", "UNABLE TO OBTAIN"`,and `"OTHER/UNKNOWN"` categories into one category called `"Other/Unknown"` to get rid of redundant data. The query then, joins prescriptions and admissions using `hadm_id` to map each drug to the patient's ethnicity. Then, assuming that `dose_val_rx` is the amount of a drug used, the next query sums up this value while grouping by ethnicity and drug for each group to calculate the total amount of each drug used by ethnicity. Finally, `ROW_NUMBER()` is used to get the top usage in each ethnicity group based on the previously calculated total dose value.


```python
import matplotlib.pyplot as plt
import pandas as pd

amount_by_ethnicity_df["total_amount"] = pd.to_numeric(amount_by_ethnicity_df["total_amount"], errors='coerce')
amount_by_ethnicity_df = amount_by_ethnicity_df.sort_values(by='total_amount', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(amount_by_ethnicity_df["ethnicity"], amount_by_ethnicity_df["total_amount"])

for bar, drug in zip(bars, amount_by_ethnicity_df["drug"]):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 2000, drug, ha='center', va='bottom', fontsize=9, rotation=90)


plt.title("Top Drug Dosage by Ethnicity (Total Amount)", fontsize=14)
plt.xlabel("Ethnicity", fontsize=12)
plt.ylabel("Total Drug Dosage", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```


    
![png](output_11_0.png)
    



(d) As can be seen in the bar graph above, Heparin is the most used drug among White people with a value of 427700 which is by far the highest number of doses in all the ethnicity groups. The next highest number of drug doses is 150000 for Heparin Sodium for people of Black or African American ethnicity. Puerto Rican Hispanic/Latino people use 0.9% Sodium Chloride the most. Hispanic/Latino people use 5% dextrose the most while Asians use Heparin the most. This suggests that Heparin and its variants seem to be the most commonly used drugs across ethnicities.

#### Question 2


```python
conn.sql(
    """
    CREATE OR REPLACE TABLE patient_age AS
    SELECT 
        p.subject_id,
        a.hadm_id,
        DATE_PART('year', AGE(a.admittime, p.dob)) AS age
    FROM patients p
    JOIN admissions a ON p.subject_id = a.subject_id;
    """
)
```


```python
conn.sql(
    """
    CREATE OR REPLACE TABLE patient_age_grouped AS
    SELECT *,
        CASE 
            WHEN age <= 19 THEN '<=19'
            WHEN age BETWEEN 20 AND 49 THEN '20-49'
            WHEN age BETWEEN 50 AND 79 THEN '50-79'
            ELSE '>80'
        END AS age_group
    FROM patient_age;
    """
)
```


```python
conn.sql(
    """
    CREATE OR REPLACE TABLE procedures_by_age AS
    SELECT 
        pag.age_group,
        pr.icd9_code
    FROM patient_age_grouped pag
    JOIN procedures_icd pr 
        ON pag.subject_id = pr.subject_id AND pag.hadm_id = pr.hadm_id;
    """
)
```


```python
conn.sql(
    """
    CREATE OR REPLACE TABLE procedure_counts AS
    SELECT 
        age_group,
        icd9_code,
        COUNT(*) AS proc_count
    FROM procedures_by_age
    GROUP BY age_group, icd9_code;
    """
)
```


```python
conn.sql(
    """
    CREATE OR REPLACE TABLE procedures_with_names AS
    SELECT 
        pc.age_group,
        pc.icd9_code,
        dp.long_title AS procedure_name,
        pc.proc_count
    FROM procedure_counts pc
    LEFT JOIN d_icd_procedures dp ON pc.icd9_code = dp.icd9_code;
    """
)
```


```python
top_3_df = conn.sql(
    """
    SELECT *
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY age_group ORDER BY proc_count DESC) AS rn
        FROM procedures_with_names
    )
    WHERE rn <= 3
    ORDER BY 
        CASE 
            WHEN age_group = '<=19' THEN 1
            WHEN age_group = '20-49' THEN 2
            WHEN age_group = '50-79' THEN 3
            WHEN age_group = '>80' THEN 4
            ELSE 5
        END,
        rn;
    """
).df()

top_3_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_group</th>
      <th>icd9_code</th>
      <th>procedure_name</th>
      <th>proc_count</th>
      <th>rn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;=19</td>
      <td>3893</td>
      <td>Venous catheterization, not elsewhere classified</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;=19</td>
      <td>8659</td>
      <td>Closure of skin and subcutaneous tissue of oth...</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;=19</td>
      <td>0118</td>
      <td>Other diagnostic procedures on brain and cereb...</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20-49</td>
      <td>3893</td>
      <td>Venous catheterization, not elsewhere classified</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20-49</td>
      <td>966</td>
      <td>Enteral infusion of concentrated nutritional s...</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20-49</td>
      <td>5491</td>
      <td>Percutaneous abdominal drainage</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>50-79</td>
      <td>3893</td>
      <td>Venous catheterization, not elsewhere classified</td>
      <td>26</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>50-79</td>
      <td>966</td>
      <td>Enteral infusion of concentrated nutritional s...</td>
      <td>22</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>50-79</td>
      <td>9904</td>
      <td>Transfusion of packed cells</td>
      <td>13</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>&gt;80</td>
      <td>3893</td>
      <td>Venous catheterization, not elsewhere classified</td>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>&gt;80</td>
      <td>9904</td>
      <td>Transfusion of packed cells</td>
      <td>13</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>&gt;80</td>
      <td>9604</td>
      <td>Insertion of endotracheal tube</td>
      <td>8</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



(b) This series of queries first calculates the age of the patient at the time of admission by calculating the difference between the admission time and the date of birth. It then groups the patients into the required four categories by age. These two tables are joined on the `hadm_id` and `subject_id` to connect the procedure to the patient for each distinct hospital visit. The count of each procedure is calculated by age group. The next part displays the long title for the procedure instead of the procedure code to make it more readable. Finally, a row number is assigned to each procedure within age groups based on `proc_count` to find the top three procedures by grouping the rows by `age_group`.


```python
import matplotlib.pyplot as plt
import pandas as pdplt
import seaborn as sns

age_order = ['<=19', '20-49', '50-79', '>80']

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_3_df,
    x='age_group',
    y='proc_count',
    hue='procedure_name',
    order=age_order
)

plt.title('Top 3 Procedures by Age Group', fontsize=14)
plt.xlabel('Age Group')
plt.ylabel('Number of Procedures')
plt.legend(title='Procedure Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

```


    
![png](output_21_0.png)
    


(d) From the graph above, we can see that Venous catheterization is the most commonly performed procedure across all the age groups. For older age groups, above the age of 50, Transfusion of packed cells and Enteral infusion of concentrated nutritional substances appear consistently in the top three procedures. The number of procedures as well as the complexity of procedures performed increases significantly with age and is the highest in the 50-79 age group. This could also be due to the fact that there are lesser patients in the >80 category. For the age group below 19, the other two procedures with very low counts are Closure of skin and subcutaneous tissue of other and Repair of vertebral fracture indicating that there are very few patients below the age of 19 who underwent procedures and the ones who did had likely gotten into accidents. 

#### Question 3


```python
conn.sql("""
CREATE OR REPLACE TABLE icu_stays_duration AS
SELECT
    ie.subject_id,
    ROUND(DATE_DIFF('hour', ie.intime, ie.outtime), 2) AS icu_hours
FROM icustays ie
WHERE intime IS NOT NULL AND outtime IS NOT NULL;
""")
```


```python
conn.sql("""
CREATE OR REPLACE TABLE icu_stays_with_demo AS
SELECT 
    ie.subject_id,
    ie.hadm_id,
    ROUND(DATE_DIFF('hour', ie.intime, ie.outtime), 2) AS icu_hours,
    p.gender,
    LOWER(TRIM(a.ethnicity)) AS ethnicity
FROM icustays ie
JOIN patients p ON ie.subject_id = p.subject_id
JOIN admissions a ON ie.hadm_id = a.hadm_id
WHERE ie.intime IS NOT NULL AND ie.outtime IS NOT NULL;
""")
```


```python
conn.sql("""
CREATE OR REPLACE TABLE icu_cleaned_ethnicity AS
SELECT
    *,
    CASE
        WHEN ethnicity LIKE '%white%' THEN 'White'
        WHEN ethnicity LIKE '%black%' THEN 'Black'
        WHEN ethnicity LIKE '%hispanic%' THEN 'Hispanic'
        WHEN ethnicity LIKE '%asian%' THEN 'Asian'
        ELSE 'Other/Unknown'
    END AS ethnicity_group
FROM icu_stays_with_demo;
""")
```


```python
conn.sql("""
SELECT
    gender,
    COUNT(*) AS num_stays,
    ROUND(AVG(icu_hours), 2) AS avg_hours,
    ROUND(MEDIAN(icu_hours), 2) AS median_hours
FROM icu_cleaned_ethnicity
GROUP BY gender;
""")
```




    ┌─────────┬───────────┬───────────┬──────────────┐
    │ gender  │ num_stays │ avg_hours │ median_hours │
    │ varchar │   int64   │  double   │    double    │
    ├─────────┼───────────┼───────────┼──────────────┤
    │ F       │        63 │    132.98 │         58.0 │
    │ M       │        73 │     84.32 │         46.0 │
    └─────────┴───────────┴───────────┴──────────────┘




```python
conn.sql("""
SELECT
    ethnicity_group,
    COUNT(*) AS num_stays,
    ROUND(AVG(icu_hours), 2) AS avg_hours,
    ROUND(MEDIAN(icu_hours), 2) AS median_hours
FROM icu_cleaned_ethnicity
GROUP BY ethnicity_group;
""")
```




    ┌─────────────────┬───────────┬───────────┬──────────────┐
    │ ethnicity_group │ num_stays │ avg_hours │ median_hours │
    │     varchar     │   int64   │  double   │    double    │
    ├─────────────────┼───────────┼───────────┼──────────────┤
    │ Other/Unknown   │        17 │    131.24 │         64.0 │
    │ Asian           │         2 │      93.0 │         93.0 │
    │ Black           │         7 │    184.29 │         95.0 │
    │ White           │        92 │     99.11 │         47.5 │
    │ Hispanic        │        18 │     94.89 │         63.5 │
    └─────────────────┴───────────┴───────────┴──────────────┘



(b) This query first calculates the duration of the ICU stay for each patient, in hours. It uses the `DATE_DIFF` function to subtract the intime from the outtime to calculate the number of hours a patient stays in the ICU. To analyze if there is a difference in gender and ethnicity, the duration table is joined with `patients` and `admissions` tables and cleaned by removing leading and trailing white spaces. Finally, the data is grouped by gender and ethnicity and the average ICU stay for each gender and each ethnicity is calculated. I calculated both the average and the median because the median is a better indicator if outliers are present. If some patients have really long stays, the mean can get skewed.

(d) We can see from the table above that females have a higher average ICU stay duration compared to males. Females spend around 132.98 hours on average with the median being 58 hours, while males spend around 84.32 hours on average with the median being 46 hours. The average and the median is higher for females.
As for the ethnicity, White patients had the highest number of ICU stays by far but had the lowest average ICU stay duration of 99 hours. Black patients had the highest average ICU stay duration of 184 hours with a suprisingly high median of 95 hours, but had fewer ICU stays compared to White patients and other ethnicities. Asians had the least number of ICU stays with an average length of 93 hours. The Hispanic patients had a similar average length of 94 hours but a median of 63.5 hours. A significant number of ICU stays were ambiguous ethnicity patients but they had an average ICU stay length of 132 hours with a median of 64 hours.  

### Part II


```python
%pip install cassandra-sigv4
```

    Requirement already satisfied: cassandra-sigv4 in c:\users\siddh\appdata\local\programs\python\python311\lib\site-packages (4.0.2)Note: you may need to restart the kernel to use updated packages.
    
    Requirement already satisfied: cassandra-driver in c:\users\siddh\appdata\local\programs\python\python311\lib\site-packages (from cassandra-sigv4) (3.29.2)
    Requirement already satisfied: boto3 in c:\users\siddh\appdata\local\programs\python\python311\lib\site-packages (from cassandra-sigv4) (1.38.5)
    Requirement already satisfied: six in c:\users\siddh\appdata\roaming\python\python311\site-packages (from cassandra-sigv4) (1.16.0)
    Requirement already satisfied: botocore<1.39.0,>=1.38.5 in c:\users\siddh\appdata\local\programs\python\python311\lib\site-packages (from boto3->cassandra-sigv4) (1.38.5)
    Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in c:\users\siddh\appdata\local\programs\python\python311\lib\site-packages (from boto3->cassandra-sigv4) (1.0.1)
    Requirement already satisfied: s3transfer<0.13.0,>=0.12.0 in c:\users\siddh\appdata\local\programs\python\python311\lib\site-packages (from boto3->cassandra-sigv4) (0.12.0)
    Requirement already satisfied: geomet<0.3,>=0.1 in c:\users\siddh\appdata\local\programs\python\python311\lib\site-packages (from cassandra-driver->cassandra-sigv4) (0.2.1.post1)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in c:\users\siddh\appdata\roaming\python\python311\site-packages (from botocore<1.39.0,>=1.38.5->boto3->cassandra-sigv4) (2.8.2)
    Requirement already satisfied: urllib3!=2.2.0,<3,>=1.25.4 in c:\users\siddh\appdata\roaming\python\python311\site-packages (from botocore<1.39.0,>=1.38.5->boto3->cassandra-sigv4) (2.4.0)
    Requirement already satisfied: click in c:\users\siddh\appdata\local\programs\python\python311\lib\site-packages (from geomet<0.3,>=0.1->cassandra-driver->cassandra-sigv4) (8.1.8)
    Requirement already satisfied: colorama in c:\users\siddh\appdata\roaming\python\python311\site-packages (from click->geomet<0.3,>=0.1->cassandra-driver->cassandra-sigv4) (0.4.6)


    
    [notice] A new release of pip is available: 23.1.2 -> 25.1.1
    [notice] To update, run: python.exe -m pip install --upgrade pip



```python
!curl https://certs.secureserver.net/repository/sf-class2-root.crt -O
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
    100  1468  100  1468    0     0   6760      0 --:--:-- --:--:-- --:--:--  6796



```python
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel
from cassandra.cluster import ExecutionProfile, EXEC_PROFILE_DEFAULT

from ssl import SSLContext, PROTOCOL_TLSv1_2, CERT_REQUIRED
from cassandra_sigv4.auth import SigV4AuthProvider
import boto3

# ssl setup
ssl_context = SSLContext(PROTOCOL_TLSv1_2)
ssl_context.load_verify_locations('sf-class2-root.crt')
ssl_context.verify_mode = CERT_REQUIRED

# boto3 session setup
boto_session = boto3.Session(region_name="us-east-2")  # this AWS credentials is specific to `us-east-2` region
```

    C:\Users\siddh\AppData\Local\Temp\ipykernel_9536\2585281160.py:10: DeprecationWarning: ssl.PROTOCOL_TLSv1_2 is deprecated
      ssl_context = SSLContext(PROTOCOL_TLSv1_2)



```python
auth_provider = SigV4AuthProvider(boto_session)
```


```python
#cluster setup 
ep = ExecutionProfile(consistency_level=ConsistencyLevel.LOCAL_QUORUM)

cluster = Cluster(['cassandra.us-east-2.amazonaws.com'], 
                  ssl_context=ssl_context, 
                  auth_provider=auth_provider,
                  execution_profiles={EXEC_PROFILE_DEFAULT: ep}, 
                  port=9142)  # TLS only communicates on port 9142
```


```python
session = cluster.connect()
```


```python
session.execute("SELECT keyspace_name FROM system_schema.keyspaces;")
session.set_keyspace('mimic')
```


```python
## creating the table

session.execute("""
    CREATE TABLE IF NOT EXISTS drug_amount_by_ethnicity (
        ethnicity_group TEXT,
        drug TEXT,
        total_amount DOUBLE,
        PRIMARY KEY (ethnicity_group, drug)
    );
""")
```




    <cassandra.cluster.ResultSet at 0x2750af16090>




```python
## uplaoding the data

for _, row in amount_by_ethnicity_df.iterrows():
    session.execute("""
        INSERT INTO drug_amount_by_ethnicity (ethnicity_group, drug, total_amount)
        VALUES (%s, %s, %s)
    """, (row['ethnicity'], row['drug'], float(row['total_amount'])))
```


```python
rows = session.execute("SELECT * FROM drug_amount_by_ethnicity;")
df_keyspace = pd.DataFrame(rows)
```

(c) I did not use Cassandra for aggregation but used Pandas instead because I wanted to use functions such as limit and groupby that are only supported by pandas and found it simpler to use that.


```python
## post extraction analysis using pandas

df_top_drug = (
    df_keyspace
    .sort_values(['ethnicity_group', 'total_amount'], ascending=[True, False])
    .groupby('ethnicity_group')
    .head(1)
)
df_top_drug
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ethnicity_group</th>
      <th>drug</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGN...</td>
      <td>5% Dextrose</td>
      <td>16900.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ASIAN</td>
      <td>Heparin</td>
      <td>15000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BLACK/AFRICAN AMERICAN</td>
      <td>Heparin Sodium</td>
      <td>150000.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HISPANIC OR LATINO</td>
      <td>5% Dextrose</td>
      <td>19950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HISPANIC/LATINO - PUERTO RICAN</td>
      <td>0.9% Sodium Chloride</td>
      <td>43663.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Other/Unknown</td>
      <td>Epoetin Alfa</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WHITE</td>
      <td>Heparin</td>
      <td>427700.0</td>
    </tr>
  </tbody>
</table>
</div>



(d) The data table printed above proves that the extraction produces the expected results.


```python
## creating the table

session.execute("""
    CREATE TABLE IF NOT EXISTS procedures_by_age_group (
        age_group TEXT,
        icd9_code TEXT,
        procedure_name TEXT,
        proc_count INT,
        PRIMARY KEY (age_group, icd9_code)
    );
""")
```




    <cassandra.cluster.ResultSet at 0x2750de8c3d0>




```python
## uploading the data

for _, row in df_top_3.iterrows():
    session.execute("""
        INSERT INTO procedures_by_age_group (age_group, icd9_code, procedure_name, proc_count)
        VALUES (%s, %s, %s, %s)
    """, (
        row['age_group'],
        row['icd9_code'],
        row['procedure_name'],
        int(row['proc_count'])
    ))

```


```python
rows = session.execute("SELECT * FROM procedures_by_age_group;")
df_cassandra = pd.DataFrame(rows)
```

(c) For this question as well, pandas was used to sort and group the data and perform aggregation instead of cassandra because I wanted to use pandas specific functions.


```python
## post extraction analysis using pandas

df_top3 = (
    df_cassandra
    .sort_values(['age_group', 'proc_count'], ascending=[True, False])
    .groupby('age_group')
    .head(3)
)
```


```python
print(df_top3)
```

       age_group icd9_code  proc_count  \
    0      20-49      3893           8   
    2      20-49       966           7   
    1      20-49      5491           6   
    10     50-79      3893          26   
    11     50-79       966          22   
    12     50-79      9904          13   
    8       <=19      3893           3   
    9       <=19      8659           2   
    7       <=19      0118           1   
    4        >80      3893          19   
    6        >80      9904          13   
    5        >80      9604           8   
    
                                           procedure_name  
    0    Venous catheterization, not elsewhere classified  
    2   Enteral infusion of concentrated nutritional s...  
    1                     Percutaneous abdominal drainage  
    10   Venous catheterization, not elsewhere classified  
    11  Enteral infusion of concentrated nutritional s...  
    12                        Transfusion of packed cells  
    8    Venous catheterization, not elsewhere classified  
    9   Closure of skin and subcutaneous tissue of oth...  
    7   Other diagnostic procedures on brain and cereb...  
    4    Venous catheterization, not elsewhere classified  
    6                         Transfusion of packed cells  
    5                      Insertion of endotracheal tube  


(d) As can be seen by the table above, the results are as expected and match the duckDB results.


```python
## creating the table

session.execute("""
    CREATE TABLE IF NOT EXISTS icu_stay_summary_v2 (
        subject_id TEXT,
        hadm_id TEXT,
        icu_hours DOUBLE,
        gender TEXT,
        ethnicity_group TEXT,
        PRIMARY KEY ((subject_id), hadm_id)
    );
""")
```




    <cassandra.cluster.ResultSet at 0x2750c9363d0>




```python
df_icu_summary = conn.sql("""
    SELECT 
        subject_id,
        hadm_id,
        icu_hours,
        gender,
        ethnicity_group
    FROM icu_cleaned_ethnicity;
""").df()
```


```python
## uplaoding the data

for _, row in df_icu_summary.iterrows():
    session.execute("""
        INSERT INTO icu_stay_summary_v2 (subject_id, hadm_id, icu_hours, gender, ethnicity_group)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        str(row['subject_id']),
        str(row['hadm_id']),
        float(row['icu_hours']),
        row['gender'],
        row['ethnicity_group']
    ))
```


```python
rows = session.execute("SELECT * FROM icu_stay_summary_v2;")
df_icu_keyspace = pd.DataFrame(rows)
```


```python
## post extraction analysis using pandas

gender_summary = (
    df_icu_keyspace
    .groupby('gender')['icu_hours']
    .agg(num_stays='count', avg_hours='mean', median_hours='median')
    .round(2)
    .reset_index()
)

ethnicity_summary = (
    df_icu_keyspace
    .groupby('ethnicity_group')['icu_hours']
    .agg(num_stays='count', avg_hours='mean', median_hours='median')
    .round(2)
    .reset_index()
)
```

(c) I used pandas for the post extraction analysis because I found it easier to group and summarize data using the pandas mean and median functions - that isn't possible directly in cassandra.


```python
print(gender_summary)
print(ethnicity_summary)
```

      gender  num_stays  avg_hours  median_hours
    0      F         59     137.07          58.0
    1      M         70      86.80          46.5
      ethnicity_group  num_stays  avg_hours  median_hours
    0           Asian          2      93.00          93.0
    1           Black          7     184.29          95.0
    2        Hispanic         17      95.12          62.0
    3   Other/Unknown         17     131.24          64.0
    4           White         86     102.78          48.0


(d) The results for this question are for the most part as expected. However, we see that there is a slight difference in the average hours spent. These minor disrepancies could be due to rounding differences, precision for floating point numbers, and missing rows during uplaoding to Cassandra. Since these are very minor differences, I think they will not affect the overall interpretation of the results.
