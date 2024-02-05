# Use Nginx to serve the static content
FROM nginx:stable-alpine
COPY . /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
