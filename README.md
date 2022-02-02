# Cotton_Yield_Estimation

This repository contains basic files and test images to run our web app for cotton yield estimation. The web app uses Flask and gunicorn to run a SVM pixel classifier to estimate cotton yield from 2D images. A Docker container image of the app is available in the repository javirodsan/yieldestimation:1.0 on Docker Hub. The image uses the Docker image taxfix/opencv-python (https://hub.docker.com/r/taxfix/opencv-python) as the base image. The additional required libraries to run the web app are installed automatically.

## Running the container

You can download the container image to your local machine by pulling it from the remote repository:

```
docker pull javirodsan/yieldestimation:1.0
```

Or you can run it directly from the remote repository:

```
docker run -ti -p 8080:8080 javirodsan/yieldestimation:1.0
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

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
