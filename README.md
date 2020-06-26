# Matching-Predicting-Scientific-Variables
This project contains a Python RESTful webservice which takes free text terms and returns suggested scientific vocabulary terms. 

The aim of this is to help those working with these terms identify which standard terms they should use as labels, or to determine the meaning of a (potentially ambiguous, confusing) label written by someone else. 

Term predictions are given as a list as 
<ul>
  <li> a) there may be multiple possible matches for a term, and </li>
  <li>b) to account for some inaccuracy.</li>
 </ul>

The vocabulary terms include observable properties and units, most of which can be found at http://registry.it.csiro.au/def/environment/_property and http://registry.it.csiro.au/def/qudt/1.1/_qudt-unit.

## Run
This implementation uses docker containers. 

Install Docker and execute: <code>  docker-compose up </code>

Once both containers are running, the app can be accessed at http://localhost:5000.
