FROM node:20-alpine

WORKDIR /app

COPY backend/package.json backend/package.json
COPY backend/src backend/src

RUN cd backend && npm install

WORKDIR /app/backend
EXPOSE 3001

ENV PORT=3001
CMD ["npm", "run", "start"]


