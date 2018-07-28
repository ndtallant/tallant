# American Community Survey

## Repository Structure
- TBA

[New Census API Changes (July 18)](https://www.census.gov/content/dam/Census/data/developers/acs/acs-data-variables-guide.pdf?eml=gd&utm_medium=email&utm_source=govdelivery)

[For when you forget how to query stuff.](https://www.census.gov/data/developers/guidance/api-user-guide/query-examples.html)
[For when you forget anything else](https://censusreporter.org/topics/getting-started/)

Start your query with the host name: https://api.census.gov/data
Add the data year to the URL; e.g., 2013
https://api.census.gov/data/2013

Add the dataset name acronym, which is available here: https://api.census.gov/data.html; e.g., acs1
https://api.census.gov/data/2013/acs1

This is the base URL for this dataset.

Start your query with a ? and add variables starting with a get clause `get=`. 
In this dataset, the variable called NAME will provide the geographic name you are using to limit your search, along with your numerical data. 
Use a comma to separate this variable from the variable designating the Hmong population; e.g., 
`?get=NAME,B02015_009E,B02015_009M.`
(A full list of ACS 1-Year geographies and variables  is available here: api.census.gov/data/2013/acs1/variables.html)
`https://api.census.gov/data/2013/acs1?get=NAME,B02015_009E,B02015_009M`

Add geography using a predicate clause starting with an ampersand (&) to separate it from your get clause and then a for followed by an in clause, if needed; e.g., &for=state. Because we are looking for information in all the states, add a wildcard (:\*) to indicate all values; e.g., state:\* 
(A full list is available here: api.census.gov/data/2013/acs1/geography.html)
`https://api.census.gov/data/2013/acs1?get=NAME,B02015_009E,B02015_009M&for=state:*`

If you are using a key, insert `&key=` followed by your key code at the end of your search URL:
`https://api.census.gov/data/2013/acs1?get=NAME,B02015_009E,B02015_009M&for=state:*&key=your key here`
