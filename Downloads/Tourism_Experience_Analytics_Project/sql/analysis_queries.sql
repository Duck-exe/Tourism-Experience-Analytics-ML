-- Tourism Experience Analytics: stakeholder-oriented SQL queries

-- 1) Dataset KPIs
SELECT
    COUNT(*) AS transactions,
    COUNT(DISTINCT UserId) AS users,
    COUNT(DISTINCT AttractionId) AS historical_attractions,
    ROUND(AVG(Rating), 3) AS average_rating
FROM tourism_analytics;

-- 2) Visit-mode distribution
SELECT VisitModeName, COUNT(*) AS visits, ROUND(AVG(Rating), 3) AS avg_rating
FROM tourism_analytics
GROUP BY VisitModeName
ORDER BY visits DESC;

-- 3) Top attractions by visit count
SELECT AttractionId, Attraction, AttractionTypeLabel,
       COUNT(*) AS visits, ROUND(AVG(Rating), 3) AS avg_rating
FROM tourism_analytics
GROUP BY AttractionId, Attraction, AttractionTypeLabel
ORDER BY visits DESC
LIMIT 15;

-- 4) Highest-rated sufficiently visited attractions
SELECT AttractionId, Attraction, COUNT(*) AS visits, ROUND(AVG(Rating), 3) AS avg_rating
FROM tourism_analytics
GROUP BY AttractionId, Attraction
HAVING COUNT(*) >= 100
ORDER BY avg_rating DESC, visits DESC
LIMIT 15;

-- 5) User continent trends
SELECT UserContinent, COUNT(*) AS visits, ROUND(AVG(Rating), 3) AS avg_rating
FROM tourism_analytics
GROUP BY UserContinent
ORDER BY visits DESC;

-- 6) Attraction type performance
SELECT AttractionTypeLabel, COUNT(*) AS visits, ROUND(AVG(Rating), 3) AS avg_rating
FROM tourism_analytics
GROUP BY AttractionTypeLabel
ORDER BY visits DESC;

-- 7) Yearly tourism trend
SELECT VisitYear, COUNT(*) AS visits, ROUND(AVG(Rating), 3) AS avg_rating
FROM tourism_analytics
GROUP BY VisitYear
ORDER BY VisitYear;

-- 8) Monthly seasonality
SELECT VisitMonth, COUNT(*) AS visits, ROUND(AVG(Rating), 3) AS avg_rating
FROM tourism_analytics
GROUP BY VisitMonth
ORDER BY VisitMonth;

-- 9) Popular destination countries in the historical transaction set
SELECT AttractionCountry, COUNT(*) AS visits, ROUND(AVG(Rating), 3) AS avg_rating
FROM tourism_analytics
GROUP BY AttractionCountry
ORDER BY visits DESC;

-- 10) High-level customer segmentation by region and visit mode
SELECT UserRegion, VisitModeName, COUNT(*) AS visits, ROUND(AVG(Rating), 3) AS avg_rating
FROM tourism_analytics
GROUP BY UserRegion, VisitModeName
ORDER BY visits DESC;
