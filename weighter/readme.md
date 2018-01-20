# Weighter: Weight Tracking Application Using Slack and Google Sheets

Weighter (name may need to be revised) is a hacky method for tracking
weight using Slack as a front end and Google Sheets as a database. 
Users enter weights as a Slack message which is then uploaded to a Google
Sheet using Zapier. Then a Python script reads the Google Sheet,
analyzes the results and sends messages back to Slack.

Once the application is set up, users only have to use Slack for 
entering weights and viewing results. 
