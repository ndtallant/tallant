# American Community Survey

## Example Query
```
https://api.census.gov/data/2015/acs/acs5?get=NAME,B01001_001E,B01001_001M&for=county:017&in=state:24
```

## If you forgot
- [For when you forgot new census API changes (July 18)](https://www.census.gov/content/dam/Census/data/developers/acs/acs-data-variables-guide.pdf?eml=gd&utm_medium=email&utm_source=govdelivery)
- [For when you forget how to query stuff.](https://www.census.gov/data/developers/guidance/api-user-guide/query-examples.html)
- [For when you forget how to find detail tables.](http://api.census.gov/data/2013/acs/acs1/variables.html)
- [For when you forget geographies.](http://api.census.gov/data/2013/acs/acs1/geography.html)
- [For when you forget anything else.](https://censusreporter.org/topics/getting-started/)

## TLDR
- Start your query with the host name: `https://api.census.gov/data`
- Add "acs" and the year: `https://api.census.gov/data/2013/acs/`
- Add the dataset name acronym, which is available [here](https://api.census.gov/data.html):`https://api.census.gov/data/2013/acs/acs1`
- Start your query with a `?` and add variables starting with a get clause `get=`.
- The variable called NAME will provide the geographic name you are using to limit your search.
- Add geography using `&for=<geography>:<subset or *>`
- If you are using a key, insert `&key=` followed by your key code at the end of your search URL:
- Result: `https://api.census.gov/data/2013/acs/acs1?get=NAME,B02015_009E,B02015_009M&for=state:*&key=your key here`
