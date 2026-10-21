FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY *.go metrics_contract.txt ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /gpuwatch-exporter .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /gpuwatch-exporter /gpuwatch-exporter
EXPOSE 9400
ENTRYPOINT ["/gpuwatch-exporter"]
