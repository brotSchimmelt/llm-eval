terraform {
  required_providers {
    lambdalabs = {
      source  = "elct9620/lambdalabs"
      version = "~> 0.1"
    }
  }
}

provider "lambdalabs" {
  api_key = var.lambdalabs_api_key
}

resource "lambdalabs_instance" "gpu_instance" {
  # us-east-1, us-south-1
#   region_name        = "us-south-1"
  region_name        = "us-east-1"
  # gpu_1x_a6000, gpu_1x_a100_sxm4
#   instance_type_name = "gpu_1x_a6000"
  instance_type_name = "gpu_1x_a100_sxm4"
  ssh_key_names      = [var.ssh_key_name]
  name               = var.machine_name

}

variable "lambdalabs_api_key" {
  description = "API key for Lambda Labs"
  type        = string
  sensitive   = true
}

variable "ssh_key_name" {
  description = "Name of the SSH key stored in Lambda Labs"
  type        = string
}

variable "machine_name" {
  description = "Name of the machine"
  type        = string
}
