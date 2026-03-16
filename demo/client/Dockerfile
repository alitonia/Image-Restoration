FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Set Vite to listen on all interfaces
ENV VITE_API_URL=http://localhost:5000

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
