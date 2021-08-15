from google_images_download import google_images_download

response = google_images_download.googleimagesdownload() 

arguments = {
    "keywords":"Wildschwein",
    "format":"jpg"
}
response.download(arguments)

response = google_images_download.googleimagesdownload() 

arguments = {
    "keywords":"멧돼지",
    "format":"jpg"
}
response.download(arguments)

response = google_images_download.googleimagesdownload() 

arguments = {
    "keywords":"Sanglier",
    "format":"jpg"
}
response.download(arguments)

response = google_images_download.googleimagesdownload() 

arguments = {
    "keywords":"Cinghiale",
    "format":"jpg"
}
response.download(arguments)

response = google_images_download.googleimagesdownload() 

arguments = {
    "keywords":"野豬",
    "format":"jpg"
}
response.download(arguments)