-- ============================================================
-- 02_build_table.sql
-- Builds the modelling tables from the public CFPB snapshot
-- (bigquery-public-data, data to 2023-03-23)
-- ============================================================

-- 1. Credit card complaints with narratives, NEW issue taxonomy only.
--    The CFPB changed its product and issue categories in April 2017.
--    Old-taxonomy complaints (product = 'Credit card', 18,838 rows) are
--    excluded to avoid contradictory labels. Result: 81,996 rows, 15 issues.
CREATE OR REPLACE TABLE `cfpb-nlp.cfpb.cc_complaints` AS
SELECT
  complaint_id,
  date_received,
  product,
  subproduct,
  issue,
  subissue,
  consumer_complaint_narrative AS narrative,
  company_name,
  state,
  submitted_via,
  company_response_to_consumer,
  timely_response
FROM `bigquery-public-data.cfpb_complaints.complaint_database`
WHERE consumer_complaint_narrative IS NOT NULL
  AND product = 'Credit card or prepaid card'
  AND subproduct IN ('General-purpose credit card or charge card',
                     'Store credit card');

-- Check: expect 81,996 rows, first date ~Apr 2017, last 2023-03-23, 15 issues
SELECT COUNT(*) AS n, MIN(date_received) AS first_date,
       MAX(date_received) AS last_date, COUNT(DISTINCT issue) AS n_issues
FROM `cfpb-nlp.cfpb.cc_complaints`;

-- 2. Modelling table: 11 labels + time-based split.
--    5 small credit-report issues merged into 'Credit reporting' (2,773).
--    'Incorrect information on your report' (2,158) kept as its own class.
--    Split: train 2017-2020 (45,130), valid 2021 (14,658), test 2022-23 (22,208)
CREATE OR REPLACE TABLE `cfpb-nlp.cfpb.cc_model` AS
SELECT
  complaint_id,
  date_received,
  narrative,
  issue,
  CASE
    WHEN issue IN (
      "Problem with a credit reporting company's investigation into an existing problem",
      'Improper use of your report',
      'Credit monitoring or identity theft protection services',
      'Unable to get your credit report or credit score',
      'Problem with fraud alerts or security freezes'
    ) THEN 'Credit reporting'
    ELSE issue
  END AS label,
  CASE
    WHEN date_received < '2021-01-01' THEN 'train'
    WHEN date_received < '2022-01-01' THEN 'valid'
    ELSE 'test'
  END AS split,
  company_name,
  state,
  submitted_via,
  company_response_to_consumer
FROM `cfpb-nlp.cfpb.cc_complaints`;

-- Check: class counts per split (expect 33 rows = 3 splits x 11 labels)
SELECT split, label, COUNT(*) AS n
FROM `cfpb-nlp.cfpb.cc_model`
GROUP BY split, label
ORDER BY split, n DESC;

-- 3. Exploration
-- Complaints per year (2023 is partial: data ends 2023-03-23)
SELECT EXTRACT(YEAR FROM date_received) AS year, COUNT(*) AS n
FROM `cfpb-nlp.cfpb.cc_model`
GROUP BY year ORDER BY year;

-- Narrative length in words (median ~167; ~15% exceed BERT's ~380-word limit)
SELECT
  APPROX_QUANTILES(ARRAY_LENGTH(SPLIT(narrative, ' ')), 10) AS word_count_deciles,
  AVG(ARRAY_LENGTH(SPLIT(narrative, ' '))) AS avg_words
FROM `cfpb-nlp.cfpb.cc_model`;

-- 4. Export for modelling (saved to Google Drive as data/cc_model.csv)
SELECT complaint_id, date_received, narrative, issue, label, split
FROM `cfpb-nlp.cfpb.cc_model`;
