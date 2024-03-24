# Terraform infrastructure-as-code script for creating an AWS EC2 virtual machine
# Creates a new security group, a new key pair, and a new EC2 Ubuntu virtual machine in AWS
#
# Based on tutorial: https://www.middlewareinventory.com/blog/terraform-aws-example-ec2/
# Install Terraform first: https://learn.hashicorp.com/tutorials/terraform/install-cli
# You probably also want to install the aws cli tools: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
#
# Steps from scratch:
# - https://csiro.awsapps.com/start#/ or https://console.aws.amazon.com
# - Either create a new user:
#      IAM > Users > Add User > Create Group > select AdministratorAccess (or whatever is appropriate) > Create > ... > get user's access key name and secret for steps below
#      Save credentials in a new ~/.aws/credentials file (see https://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html for details). Make a new profile if needed.
#      IAM > Groups > Create Group > add EC2 and RDS policies > Add user to new group
#   Or ask an AWS admin get details of an existing user with appropriate access.
#
#   Get your Virtual Private Cloud (VPC) ID (or create a new one).  This is just a private network for common access by cloud resources.
#     VPC > Your VPCs > copy VPC ID
#     Paste into `vpc` variable below
#
# Then Deploy:
#   cd scripts # or dir with main.tf
#   export AWS_ACCESS_KEY_ID=*****
#   export AWS_SECRET_ACCESS_KEY=*****
#   terraform init # Install terraform locally
#   terraform apply # create infrastructure, takes a few minutes
#   terraform refresh # make sure we get the most recent IP etc
#   terraform output # get the IP address of the EC2 instance
#   terraform output -raw private_key > terraform_key.pem # get the generated private key for the EC2 instance
#   notepad++ terraform_key.pem # (Windows only?) open terraform_key.pem in good text editor and change encoding to UTF8
#   ssh -i terraform_key.pem ubuntu@<IP address> -p 22223 #Open SSH session to server
#   ...do stuff on server, such as set a password:
#     sudo su - # switch to root
#     passwd ubuntu # set password for ubuntu user
#   terraform destroy # Remove all created infrastructure when you're finished using it.

# define all the resource names that we are going to be using within the Terraform configuration
variable "awsprops" {
  type    = map(string)
  default = {
    profile       = "your-aws-profile-name" # name of the profile/credentials in ~/.aws/credentials to use, remove for default profile.
    region        = "ap-southeast-2" # select which region you want to deploy into
    vpc           = "vpc-08cc722cbbd1f3b8e" # copy from: VPC > Your VPCs > copy VPC ID
    subnet        = "subnet-00526b93353f11ab3" # copy from VPC > Subnets
    #      ami = "ami-0e040c48614ad1327" # code for the Amazon Machine Image.  Select this from https://ap-southeast-2.console.aws.amazon.com/ec2/v2/home?region=ap-southeast-2#LaunchInstances, or leave commented out to just use the most recent Ubuntu AMI (see below)
    instance_type = "t2.micro"
    # 'hardware' (and pricing) configuration to use for hosting the VM. See https://ap-southeast-2.console.aws.amazon.com/ec2/v2/home?region=ap-southeast-2#LaunchInstances.
    publicip      = true # whether to allocate a public IP address
    secgroupname  = "IAC-Sec-Group"
    server_name   = "Test-Server-01"
  }
}

# create a new SSH key pair - see https://stackoverflow.com/questions/49743220/how-do-i-create-an-ssh-key-in-terraform
resource "tls_private_key" "example" {
  algorithm = "RSA"
  rsa_bits  = 4096
}
# install new key pair into AWS
resource "aws_key_pair" "generated_key" {
  key_name   = "terraform-generated ssh key"
  public_key = tls_private_key.example.public_key_openssh
}

# Tell terraform to target AWS
provider "aws" {
  region = lookup(var.awsprops, "region")
  profile = lookup(var.awsprops, "profile")
}

#  Security Group with inbound and outbound firewall rules.
resource "aws_security_group" "project-iac-sg" {
  name        = lookup(var.awsprops, "secgroupname")
  description = lookup(var.awsprops, "secgroupname")
  vpc_id      = lookup(var.awsprops, "vpc")

  // To Allow SSH Transport map custom port (need to manually change SSHd port after creation)
  ingress {
    from_port   = 22223
    protocol    = "tcp"
    to_port     = 22223 # non-standard SSH port so aws/csiro security checks don't remove this rule
    cidr_blocks = [
      # See https://confluence.csiro.au/display/IMT/CSIRO+IP+Address+Ranges (and https://ipinfo.io/AS6262)
      "152.83.0.0/16",  #  ACT Internal Australian Capital Territory based devices
      "138.194.0.0/16", #  VIC Internal Victoria based devices
      "130.155.0.0/16", #  NSW Internal New South Wales based devices
      "140.253.0.0/16", #  QLD Internal Queensland based devices
      "140.79.0.0/16",  #  TAS Internal Tasmania based devices
      "130.116.0.0/16", #  WA Internal Western Australia based devices
      "144.110.0.0/16", #  SA/NT Internal South Australia & North Territory based devices
#      "150.229.0.0/16", #  All States External This includes guest wireless users (ie non CSIRO endpoints) so do not trust this unless you know what you are doing
#      "146.118.0.0/16", #  Pawsey Internal (Pawsey) This includes non-CSIRO users on the NIMBUS cloud so do not trust this unless you know what you are doing
      "2405:b000::/32"	# All of CSIRO IPv6
    ]
  }

  lifecycle {
    create_before_destroy = true
    # create the replacement resources first before destroying the live ones. this way we reduce downtime
  }
}

# Find the most recent AMD64 Ubuntu server image AMI code.
data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-*-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

resource "aws_instance" "project-iac" {
  ami                         = "${data.aws_ami.ubuntu.id}" # Use AMI code from search above
  instance_type               = lookup(var.awsprops, "instance_type")
  subnet_id                   = lookup(var.awsprops, "subnet")
  associate_public_ip_address = lookup(var.awsprops, "publicip")
  key_name                    = aws_key_pair.generated_key.key_name # use new generated key

  # Custom shell commands to run after VM creation - change SSHd port
  user_data = <<-EOL
    #!/bin/bash -xe
    sudo echo "Port 22223" >> /etc/ssh/sshd_config
    sudo systemctl reload sshd
  EOL

  vpc_security_group_ids = [
    aws_security_group.project-iac-sg.id
  ]
  root_block_device {
    delete_on_termination = true
    iops                  = 3000
    volume_size           = 50
    volume_type           = "gp3"
  }

  # give some meaningful tags for management and future identification
  tags = {
    Name        = lookup(var.awsprops, "server_name")
    Environment = "DEV"
    OS          = "UBUNTU"
    Managed     = "IAC"
  }
  depends_on = [aws_security_group.project-iac-sg]
}

output "ec2instance" {
  value = aws_instance.project-iac.public_ip
}

output "private_key" {
  value     = tls_private_key.example.private_key_pem
  sensitive = true
}
