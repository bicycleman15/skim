### Task Instructions

You are given a Wikipedia article, specifically the title and body text of the article. Your job is to generate immediate Wikipedia categories as well as the Wikipedia categories for the generated immediate Wikipedia categories using the information provided in the body text of the article. 

When generating Wikipedia categories, make sure to follow below mentioned guidelines for a valid Wikipedia category:

1. Relevance: The categories you generate should be directly relevant to the topic of the Title. They should describe key aspects of the subject matter. For example, if given an Title about dogs, relevant categories might include "Mammals," "Pets," "Animal Behavior," or "Dog Breeds." Categories like "Astronomy" or "Cooking" would not be relevant to a dog-related Title.

2. Specificity: Wikipedia's category system is organized hierarchically, with broader categories containing more specific subcategories. Try to generate categories that aim to place the Title in the most specific category that applies.

First generate the immediate Wikipedia categories which are directly relevant to the article. After that, generate Wikipedia categories for the earlier generated immediate Wikipedia categories. Make sure to follow the guidelines that are mentioned above. 

Below are some examples that should provide more clarity about the task.

Example 1:

### Wikipedia Title
Sellankandal

### Wikipedia article begins
Sellankandal is a village situated 10 km inland from coastal Puttalam city in the North Western Province of Sri Lanka. It is the primary settlement of people of Black African descent in Sri Lanka called Kaffirs who until the 1930s spoke a Creole version of Portuguese. Most villages speak Sinhala and are found throughout the country as well as in the Middle East. The Baila type of music, very popular in Sri Lanka since the 1980s, originated centuries ago among this 'kaffir' community. They however complain that they benefitted very little from the popularity of the Baila music.
### Wikipedia article ends

### Task Output
#### Immediate Wikipedia Categories
Populated places in North Western Province, Sri Lanka
Populated places in Puttalam District
African diaspora in Sri Lanka

#### Wikipedia Categories for Immediate Wikipedia Categories
Populated places in North Western Province, Sri Lanka
	Populated places in Sri Lanka by province
	Geography of North Western Province, Sri Lanka
Populated places in Puttalam District
	Populated places in North Western Province, Sri Lanka
	Populated places in Sri Lanka by district
	Geography of Puttalam District
	Puttalam District
African diaspora in Sri Lanka
	African diaspora in Asia
	Ethnic groups in Sri Lanka

---------------------------

Example 2:

### Wikipedia Title
Rossana

### Wikipedia article begins
Rossana is a feminine Italian given name. Notable people with the name include:

Rossana Casale (born 1959), Italian singer
Rossana Podestà (1934–2013), Italian film actress
Rossana Lombardo (born 1962), Italian sprinter
Rossana Martini (1926–1988), Italian actress, model and beauty pageant winner
Rossana Morabito (born 1969), Italian sprinter
### Wikipedia article ends

### Task Output
#### Immediate Wikipedia Categories
Italian feminine given names
Feminine given names
Given names

#### Wikipedia Categories for Immediate Wikipedia Categories
Italian feminine given names
	Feminine given names
	European feminine given names
	Italian given names
Feminine given names
	Given names
	Feminine names
Given names
	Human names

---------------------------

Example 3:

### Wikipedia Title
Wild Turkey Strand Preserve

### Wikipedia article begins
Wild Turkey Strand Preserve is a  3,137 acre area of protected lands in Lee County, Florida. The preserve is off State Route south of Lehigh Acres. It includes part of the former Buckingham Army Airfield, a World War II-era training base.

The preserve includes flatwoods, cypress strand swamps, cypress dome swamps, freshwater marshes, wet prairies, and abandoned agricultural pasture. There is a 1.8 mile trail with boardwalks and interpretive signage. The preserve lands were acquired during the first decade of the 21st century.
### Wikipedia article ends

### Task Output
#### Immediate Wikipedia Categories
Protected areas of Lee County, Florida

#### Wikipedia Categories for Immediate Wikipedia Categories
Protected areas of Lee County, Florida
	Protected areas of Florida by county
	Geography of Lee County, Florida
	Tourist attractions in Lee County, Florida

---------------------------

Example 4:

### Wikipedia Title
Malabar Coast

### Wikipedia article begins
The Malabar Coast is the southwestern region of the Indian subcontinent. It generally refers to the western coastline of India from Konkan to Kanyakumari. Geographically, it comprises one of the wettest regions of the subcontinent, which includes the Kanara region of Karnataka, and entire Kerala.

Kuttanad, which is the point of least elevation in India, lies on the Malabar Coast. Kuttanad, also known as The Rice Bowl of Kerala, has the lowest altitude in India, and is one of the few places in the world where cultivation takes place below sea level.[3][4] The peak of Anamudi, which is also the point of highest altitude in India outside the Himalayas, lies parallel to the Malabar Coast on the Western Ghats.

The region parallel to the Malabar Coast gently slopes from the eastern highland of Western Ghats ranges to the western coastal lowland. The moisture-laden winds of the Southwest monsoon, on reaching the southernmost point of the Indian subcontinent, because of its topography, divides into two branches; the "Arabian Sea Branch" and the "Bay of Bengal Branch".[5] The "Arabian Sea Branch" of the Southwest monsoon first hits the Western Ghats,[6] making Kerala the first state in India to receive rain from the Southwest monsoon.[7][8] The Malabar Coast is a source of biodiversity in India.
### Wikipedia article ends

### Task Output
#### Immediate Wikipedia Categories
Regions of India
Regions of Kerala
Landforms of Karnataka
Landforms of Kerala
Malabar Coast
Coasts of India

#### Wikipedia Categories for Immediate Wikipedia Categories
Regions of India
	Regions of Asia by country
	Geography of India
Regions of Kerala
	Regions of India by state or union territory
	Geography of Kerala
Landforms of Karnataka
	Landforms of India by state or union territory
	Geography of Karnataka
Landforms of Kerala
	Landforms of India by state or union territory
	Geography of Kerala
Malabar Coast
	Coasts of India
Coasts of India
	Coasts of the Indian Ocean
	Coasts of Asia by country
	Landforms of India
	Coasts by country
	Water in India

---------------------------

Now perform the task for the following:

Please do not use any external knowledge to generate Wikipedia categories, the information must come only from the provided article, do not generate anything extra that is not there in the article.
**Make sure that the categories generated are actual Wikipedia Categories.**
Generate as much relevant immediate Wikipedia categories as possible.

### Wikipedia Title
{title}

### Wikipedia article begins
{content}
### Wikipedia article ends

### Task Output
