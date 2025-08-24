FROM node:20-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*


# working directory
WORKDIR /app

# install pnpm globally
RUN corepack enable && corepack prepare pnpm@latest --activate

# copy only lockfile and package.json first
COPY package.json pnpm-lock.yaml* ./

# install deps first
RUN pnpm install --frozen-lockfile

# copy project files
COPY . .

# install dependencies
# RUN pnpm install --frozen-lockfile

# build the TypeScript app
RUN pnpm build

# set environment variables
ENV NODE_ENV=production

# expose the port used by the app
EXPOSE 8080

# run the app
CMD ["node", "dist/index.js"]
