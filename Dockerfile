# This Dockerfile uses a pre-built PyTorch image as the base.
# It's an excellent starting point as it already has PyTorch and CUDA configured.
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Add build arguments for ClearML credentials.
ARG CLEARML_API_KEY
ARG CLEARML_API_SECRET
ARG CLEARML_API_HOST

# Set these arguments as environment variables in the container.
ENV CLEARML_API_KEY=${CLEARML_API_KEY}
ENV CLEARML_API_SECRET=${CLEARML_API_SECRET}
ENV CLEARML_API_HOST=${CLEARML_API_HOST}

# Install necessary build tools and libraries.
# `build-essential` includes a C compiler (gcc) and other tools.
# `gfortran` is specifically required to compile numpy from source.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gfortran && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements.txt file first to leverage Docker's build cache.
# This way, if only your application code changes, this step won't be re-run.
COPY requirements.txt .

# Install the dependencies from requirements.txt.
# The previous step ensures the build tools are available for packages
# like numpy that need to be compiled.
RUN pip install -r requirements.txt

# Copy the rest of your project files.
COPY . .

# Define the command to run your application.
CMD ["python", "main.py"]