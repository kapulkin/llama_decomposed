apt-get install -y vim
echo "adduser ubuntu"
adduser ubuntu
groups ubuntu
usermod -aG sudo ubuntu
echo "visudo"
visudo

# git config --global user.name "Your Name"
# git config --global user.email "you@example.com"

sudo -i -u ubuntu bash
