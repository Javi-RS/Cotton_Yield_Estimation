# Cotton_Yield_Estimation Containerization

This repository houses essential files and test images for containerizing our web app designed to estimate cotton yield from aerial imagery. This work is a key part of the research paper titled "Cotton Yield Estimation From Aerial Imagery Using Machine Learning Approaches," published in the journal Frontiers in Plant Science. You can find the paper [here](https://doi.org/10.3389/fpls.2022.870181).

Our web app utilizes Flask and gunicorn to operate an SVM pixel classifier, enabling accurate cotton yield estimation from 2D imagery. To streamline deployment, a Docker container image is available at [javirodsan/yieldestimation:latest](https://hub.docker.com/r/javirodsan/yieldestimation) on Docker Hub. This Docker image is based on ``python:3.8-slim-buster`` and ``OpenCV`` and automatically installs all required libraries for seamless execution of the web app.

## Running the container

You can download the container image (~880MB) to your local machine by pulling it from the remote repository:

```
docker pull javirodsan/yieldestimation:latest
```

Or you can run it directly:

```
docker run -ti -p 8080:8080 javirodsan/yieldestimation:latest
```
Note that this command also does a docker pull behind the scenes to download the image with latest tag. 

## Using the web app
You should be able to see the Flask application running on http://localhost:8080 or 127.0.0.1:8080

Select image analysis from the drop down menu and upload an image to analyze. Finally, click proess image to start the estimation of yield. Please be patient, the analysis can take some time.
____
# Authors

* Javier Rodriguez - *Graduate Research Assistant* - [BSAIL Lab (University of Georgia)](https://bsail.engr.uga.edu/)
* Changying Li - *Professor and Director* - [BSAIL Lab (University of Georgia)](https://bsail.engr.uga.edu/)
* Andrew H. Paterson - *Regents Professor and Director* - [The Plant Genome Mapping Laboratory (University of Georgia)](http://www.plantgenome.uga.edu/)

## Find us on GitHub

* [Javi-RS](https://github.com/Javi-RS)
* [UGA-BSAIL](https://github.com/UGA-BSAIL)

____
# Acknowledgments
Thanks to Rui Xu for his assistance in collecting aerial images; Tsunghan Han, Shangpeng Sun, and Jon S. Robertson for helping in the manual harvesting of the cotton bolls for ground truthing; Man Wah Aw for his assistance in image annotation; and Ellen Skelton for providing the vegetal material.

____
# License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/Javi-RS/Cotton_Yield_Estimation/blob/main/LICENSE) file for details.
