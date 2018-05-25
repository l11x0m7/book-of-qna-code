# ELK Service
[elasticsearch](https://hub.docker.com/_/elasticsearch/)
[kibana](https://hub.docker.com/_/kibana/)

![](https://camo.githubusercontent.com/ae91a5698ad80d3fe8e0eb5a4c6ee7170e088a7d/687474703a2f2f37786b6571692e636f6d312e7a302e676c622e636c6f7564646e2e636f6d2f61692f53637265656e25323053686f74253230323031372d30342d30342532306174253230382e32302e3437253230504d2e706e67)

## Installation
```
docker pull kibana:5.4.0
docker pull elasticsearch:5.4.0
docker pull mobz/elasticsearch-head:5
docker pull neo4j:3.1.4
```

## Configuration
```
cp es_config/elasticsearch.sample.yml esconfig/elasticsearch.yml
cp es_config/log4j2.sample.properties esconfig/log4j2.properties
```

Update the Configurations.

## Start
```
cd elk-service
./start-elasticsearch -d
./start-elasticsearch-head
./start-kibana -d
./start-neo4j
open YOUR_IP:5061 # for kibana
open YOUR_IP:9100 # for elasticsearch head
open YOUR_IP:9200 # for elasticsearch
```

OR 

## Run with docker compose
```
docker-compose up -d [--force-recreate]
docker-compose logs -f --tail="all" # logs
```

## Test
Open http://YOUR_IP:7474/browser/
```
merge(p:Person{name:"hain", createAt:toString(TIMESTAMP())})
```

Now, check elasticsearch-head(http://YOUR_IP:9100/), add index in kibana.

## Index
### Create Document
```
PUT /twitter/tweet/3 HTTP/1.1
Host: YOUR_IP:9200
Content-Type: application/json

{
    "user" : "kimchy",
    "post_date" : "2016-11-15T14:12:16",
    "message" : " out xxx"
}
```

**twitter** Index
**tweet** Type
**3** type id


### Search Document

```
GET /twitter/tweet/_search HTTP/1.1
Host: YOUR_IP:9200
Accept: application/json
```

```
GET /twitter/tweet/_search?q=message:out HTTP/1.1
Host: YOUR_IP:9200
```

[Search in depth](https://www.elastic.co/guide/en/elasticsearch/guide/current/search-in-depth.html)

[Dealing with Human Language](https://www.elastic.co/guide/en/elasticsearch/guide/current/languages.html#languages)

[Aggregations](https://www.elastic.co/guide/en/elasticsearch/guide/current/aggregations.html)

## Plugins
### elasticsearch-head
https://github.com/mobz/elasticsearch-head

### neo4j
https://neo4j.com/developer/elastic-search/
https://github.com/Samurais/graph-aided-search
https://graphaware.com/neo4j/2016/04/20/graph-aided-search-the-rise-of-personalised-content.html

## Destroy
```
docker-compose down
# flush-all-data.sh // delete all data
```

## Security
[How to Secure Elasticsearch and Kibana](https://www.mapr.com/blog/how-secure-elasticsearch-and-kibana)

### TL;DR
```
sudo apt-get install apache2-utils nginx -yy
sudo mkdir -p /opt/elk/
sudo htpasswd -c /opt/elk/.espasswd sysadmin
```

### nginx_ensite

```
# install nginx_ensite
cd ~
git clone https://github.com/Samurais/nginx_ensite.git
cd nginx_ensite
make install

# install nginx conf
cd ~/elk-service
cp  nginx/es.conf /etc/nginx/sites-available/elk-es.xxx.net.conf
cp  nginx/kibana.conf /etc/nginx/sites-available/elk-kibana.xxx.net.conf
# update conf
nginx_ensite elk-es.xxx.net.conf
nginx_ensite elk-kibana.xxx.net.conf
sudo service nginx reload
```


## Client
[elasticsearch.js](https://www.elastic.co/guide/en/elasticsearch/client/javascript-api/current/index.html)
[Importing data into Elasticsearch](https://gist.github.com/Samurais/0da7bcbe0cc5830b118b411596f2c171)

## Further Reading
[Elasticsearch Definitive Guide](./elasticsearch-definitive-guide-en.pdf)

## Trouble Shooting

1. [Fielddata is disabled on text fields by default](https://www.elastic.co/guide/en/elasticsearch/reference/5.0/fielddata.html)
```
PUT /chatbot/_mapping/messageinbound/ HTTP/1.1
Host: elk-es.xxx.net
Content-Type: application/json
Authorization: Basic xxxxxxxxxxxxxxx

{
  "properties": {
    "fromUserId": { 
      "type":     "text",
      "fielddata": true
    }
  }
}
```

2. Can not find timestamp field to create Kibana Index.

> that is due to elasticsearch:index mappings are mapped to un-Date type.

Set the date object with RESt API.
```
PUT /neo4j-index-node/_mapping/Person HTTP/1.1
Host: YOUR_IP:9200
Content-Type: application/json

{
	"properties":{
        "createAt": {
          "type": "date",
          "format": "epoch_millis"
        }
      }
}
```

Note, with neo4j-to-elasticsearch, use **merge(p:Person{name:"hain", createAt:toString(TIMESTAMP())})** .


# LICENSE
All Rights Reserve 2017, Hai Liang Wang.
