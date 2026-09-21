# Data Dictionary

## Supplied raw tables

### Transaction.xlsx
- `TransactionId`: unique transaction identifier.
- `UserId`: user foreign key.
- `VisitYear`: year of visit.
- `VisitMonth`: month number (1-12).
- `VisitMode`: numeric visit-mode ID.
- `AttractionId`: attraction foreign key.
- `Rating`: observed 1-5 rating.

### User.xlsx
- `UserId`: user identifier.
- `ContinentId`, `RegionId`, `CountryId`, `CityId`: user geography keys.

### City.xlsx
- `CityId`: city key.
- `CityName`: city label.
- `CountryId`: country foreign key.

### Country.xlsx
- `CountryId`: country key.
- `Country`: country label.
- `RegionId`: region foreign key.

### Region.xlsx
- `Region`: region label.
- `RegionId`: region key.
- `ContinentId`: continent foreign key.

### Continent.xlsx
- `ContinentId`: continent key.
- `Continent`: continent label.

### Mode.xlsx
- `VisitModeId`: mode key.
- `VisitMode`: Business, Couples, Family, Friends, Solo (plus placeholder row 0).

### Type.xlsx
- `AttractionTypeId`: numeric attraction-type key.
- `AttractionType`: type label.

### Item.xlsx
- `AttractionId`: attraction key.
- `AttractionCityId`: attraction city key.
- `AttractionTypeId`: attraction type key.
- `Attraction`: attraction name.
- `AttractionAddress`: address.

### Additional_Data_for_Attraction_Sites/Updated_Item.xlsx
Expanded catalog with 1,698 attraction rows. The raw `AttractionTypeId` field contains a mixture of numeric IDs and text labels. The cleaning pipeline preserves this as `AttractionTypeRaw`, parses numeric values when possible, and creates safe textual type features.

## Processed tables

### tourism_merged_clean.csv
Transaction-level analytic table. It contains raw transaction fields plus:
- `VisitModeName`
- `UserContinent`, `UserRegion`, `UserCountry`, `UserCityName`
- historical attraction name/address/type
- attraction city/country/region/continent labels
- `AttractionTypeLabel` and `AttractionTypeBroad`

### attraction_catalog_clean.csv
Expanded recommendation catalog. Important derived fields:
- `AttractionTypeRaw`: source value as text.
- `AttractionTypeNumericId`: parsed nullable numeric ID.
- `AttractionTypeLabel`: resolved official label when numeric, otherwise cleaned source label.
- `AttractionTypeBroad`: broader recommendation category (e.g., Beach, Museum, Religious, Nature).
- `HistoricalVisitCount`, `HistoricalAverageRating`: historical signals for attractions found in transaction data; zero/global fallback for unseen catalog items.

## Model feature policy

### Classification
Uses user geography, year/month, attraction identity/type/location. It **does not use `Rating`** and does not use `UserId`.

### Regression
Uses the classification features plus `VisitModeName`. It does not use `UserId`.

### Recommendation
Uses explicit `UserId`-attraction-rating histories for collaborative filtering and attraction content/location for content-based filtering.
