---- Date range and total complaints
SELECT
  MIN(date_received) AS first_date,
  MAX(date_received) AS last_date,
  COUNT(*) AS total_complaints
FROM `bigquery-public-data.cfpb_complaints.complaint_database`;

--Complaints by product
SELECT
  product,
  COUNT(*) AS complaints,
  COUNTIF(consumer_complaint_narrative IS NOT NULL) AS with_narrative
FROM `bigquery-public-data.cfpb_complaints.complaint_database`
GROUP BY product
ORDER BY complaints DESC;
