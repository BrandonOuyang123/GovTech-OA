# GovTech-OA
Due to time constraints, I implemented a very basic prototype.

Outlining my full approach:
1. Setting up of BigQuery as database
2. Host streamlit app on cloud run
3. Connect AI to BigQuery (pandas-gbq), and allow it to query any data the user is interested in, provide insights and visualisations
4. Model the data with ML such as linear regression, random forest, XGBoost etc
5. Using a chatbot with access to local environment code execution so AI can process user query, execute code, and present desired answer

For scaling:
- map out singapore HDBs, and present the visualisation of BTOs on a map with its resale pricing, expandable highlights
- More in-depth research on factors affecting pricing such as using location to calculate shortest distance to nearby amenities like supermarkets, malls and mrts
- scoping with clients to determine the use case of this product
