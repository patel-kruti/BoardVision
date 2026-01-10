FROM node:20-alpine

WORKDIR /app

COPY frontend/package.json frontend/package.json
COPY frontend/index.html frontend/index.html
COPY frontend/vite.config.js frontend/vite.config.js
COPY frontend/postcss.config.cjs frontend/postcss.config.cjs
COPY frontend/tailwind.config.cjs frontend/tailwind.config.cjs
COPY frontend/eslint.config.js frontend/eslint.config.js
COPY frontend/src frontend/src

RUN cd frontend && npm install

WORKDIR /app/frontend
EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

