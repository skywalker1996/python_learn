import smtplib
from email.mime.text import MIMEText
from email import encoders
from email.header import Header
from email.utils import parseaddr, formataddr
import smtplib

def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr(( \
        Header(name, 'utf-8').encode(), \
        addr.encode('utf-8') if isinstance(addr, unicode) else addr))

from_addr = "zjHui_hhuc@163.com"
from_password = 'hui1996'

to_addr = "zjhui@hhu.edu.cn"
smtp_addr = "smtp.163.com"

msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')
msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')
msg['From'] = _format_addr(u'Python爱好者 <%s>' % from_addr)
msg['To'] = _format_addr(u'管理员 <%s>' % to_addr)
msg['Subject'] = Header(u'来自SMTP的问候……', 'utf-8').encode()

print("creat the sever!")
sever = smtplib.SMTP(smtp_addr, 25)
print("enter the sever!")
sever.set_debuglevel(1)
sever.login(from_addr, from_password)
print("log in successfully !")
sever.sendmail(from_addr, [to_addr], msg.as_string())
sever.quit()
