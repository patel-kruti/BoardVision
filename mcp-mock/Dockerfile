FROM node:20-alpine

WORKDIR /app

COPY mcp-mock/package.json mcp-mock/package.json
COPY mcp-mock/src mcp-mock/src

RUN cd mcp-mock && npm install

WORKDIR /app/mcp-mock
EXPOSE 8080

ENV PORT=8080
CMD ["npm", "run", "start"]


