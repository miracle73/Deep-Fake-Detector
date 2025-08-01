FROM node:20-slim

# working directory
WORKDIR /app

# install pnpm globally
RUN corepack enable && corepack prepare pnpm@latest --activate

# copy project files
COPY . .

# install dependencies
RUN pnpm install --frozen-lockfile

# build the TypeScript app
RUN pnpm build

# set environment variables
ENV NODE_ENV=production

# expose the port used by the app
EXPOSE 8080

# run the app
CMD ["node", "dist/index.js"]
