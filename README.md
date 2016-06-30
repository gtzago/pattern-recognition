# pattern-recognition
Codes for the course of Pattern Recognition in 2016/1.

### Atualiza o sistema
```sh
sudo apt-get update
sudo apt-get upgrade
sudo reboot
```
### Instala pacotes de desenvolvimento
```sh
sudo apt-get install python python-dev python-setuptools python-virtualenv 
```
### Configura área de trabalho
Dentro de sua pasta de usuário, crie a seguinte estrutura de diretórios:
/desenvolvimento
/desenvolvimento/repositorio
/desenvolvimento/env

### Instala IDE e dependências
```sh
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
sudo apt-get install oracle-java8-set-default
```

Faça o download e instale o eclipse na pasta desenvolvimento

### Instala programa de gerenciamento de código
sudo apt-get install git git-gui

Solicita Acesso ao Repositório
Para acessar o repositório é necessário informar sua chave ssh ao
administrador do repositório.
Verifique qual é a sua chave através de:
terminal -> git gui
menu “Ajuda” -> “Mostrar chave ssh”.
Caso não exista nenhuma chave gerada, gere-a por meio do botão “Gerar
chave”. Deixe passphrase em branco.

### Clonagem do código pela primeira vez
```sh
cd ~desenvolvimento/repositorio
mkdir pattern-recognition
cd pattern-recognition
git init
git remote add origin <https://github.com/...........git>
git fetch origin
git merge origin/master
```

### Instala as dependências do sistema
use
```sh
sudo apt-get install ...
```
para cada pacote contido em ubuntu.txt, na pasta pattern-recognition

### Cria ambiente virtual
```sh
cd ~/desenvolvimento/env/
virtualenv --system-site-packages patrec
```

a opção (--system-site-packages) permite ao ambiente virtual utilizar os pacotes previamente instalados no sistema, necessário por causa do gtk, usado para plotar os gráficos no matplotlib.

### Ativa Ambiente virtual
```sh
source patrec/bin/activate
```

### instala dependências do projeto no ambiente virtual

```sh
source ~/desenvolvimento/env/patrec/bin/activate
pip install -r ../repositorio/pattern-recognition/requirements.txt
```

### Prepara a IDE
abra o eclipse

help -> eclipse marketplace -> find -> “pydev” -> install
window -> perspective -> open perspective -> other -> pydev
file-> new pydev project -> 
project name: pattern-recognition
directory: desenvolvimento/repositorio/pattern-recognition

please configure an interpreter before proceeding->manual config -> new ->
Interpreter Name: patrec
interpreter executable : desenvolvimento/env/patrec/bin/python2.7->ok

botao direito no projeto -> properties ->pydev-python-path->add source folder -> pattern-recognition

Formatador de código
window -> preferences -> pydev -> editor-> code analysis -> pep8.py -> warning -> ok
window -> preferences -> pydev -> editor-> code style -> code formatter -> use autopep8.py for code formatting? v
ok

dicas
ctrl+shift+o (organiza os imports)
ctrl+shift+f (formata o codigo utilizando o pep8)

### Utilizando o git
Ao iniciar o trabalho do dia, faça:
```sh
git fetch origin
git merge origin/master
```
Ao concluir uma etapa, envie as alterações:

```sh
git add -A
git commit -m “mensagem sobre as alteracoes”
git push origin master
```
