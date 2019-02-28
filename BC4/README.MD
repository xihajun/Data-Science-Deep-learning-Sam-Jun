# BC4
## SSH login [serverfault](https://serverfault.com/questions/241588/how-to-automate-ssh-login-with-password)

```{bash}
# you can skip passphrase and save the key in the default path by hit enter
ssh-keygen -t rsa -b 2048 #generating the public/private rsa key pair
# Then copy keys to the target server by using
ssh-copy-id yourusername@bc4login.acrc.bris.ac.uk
# You now can login in by using
ssh yourusername@bc4login.acrc.bris.ac.uk
```

In order to make my life easier, I write a bash command to generate automatically. Download the autokey.sh and run `./autokey.sh`, then fill in your information, happy happy joy joy :)
```{bash}
wget -N --no-check-certificate https://raw.githubusercontent.com/xihajun/vagex/master/vagex_test.sh && bash vagex_test.sh

```
