import re
import lxml.html
from lxml.html import HtmlComment


class Extractor(object):
    def __init__(self,):
        self.non_content_tag = set([
            'script','embed', 'iframe',
            'style',
        ])
        pass

    def get_encoding(self, html):
        r = r'charset=["\']?([\d\w\-]*)["\' />]?'
        m = re.search(r, html, re.I)
        if m:
            return m.groups()[0].lower()
        #TODO: detect charset using chardet
        return 'utf-8'

    def get_text(self, doc, is_parent=False):
        lines_element = []
        if is_parent:
            # is 'doc' is content-element's parent,
            # content-lines is its grandchildren
            ch = doc.getchildren()
            for c in ch:
                grandchildren = c.getchildren()
                if grandchildren:
                    lines_element.extend(grandchildren)
                else:
                    lines_element.append(c)
        else:
            lines_element = doc.getchildren()
        lines = []
        for el in lines_element:
            line = ''
            if el.text:
                line += el.text.strip()
            for ch in el.iter():
                if ch.text:
                    line += ch.text.strip()
            if line:
                lines.append(line)
        return '\n'.join(lines).encode('utf8')

    def get_content(self, html, just_content=True, with_tag=True):
        encoding = self.get_encoding(html)
        if encoding not in ['utf-8', 'utf8']:
            html = html.decode(encoding, 'ignore')
        doc = lxml.html.fromstring(html)

        # 1. find the longest text block, which is the main content
        body = doc.xpath('//body')
        if not body:
            print('no body, parse whole document tree')
            body = doc
        else:
            body = body[0]
        elements = [body]
        last_max_len = 0
        good_el = None
        while elements:
            p = elements.pop(0)
            tlen = 0
            if p.text:
                tlen += len(p.text.strip())
            for el in p.iterchildren():
                if (el.tag in self.non_content_tag or
                    isinstance(el, HtmlComment)):
                    continue
                elements.append(el)
                if el.text:
                    t = el.text.strip()
                    tlen += len(t)
                if el.tail:
                    tlen += len(el.tail.strip())
            if last_max_len and tlen > 50*last_max_len:
                print('break at: ', last_max_len, tlen)
                last_max_len = tlen
                good_el = p
                break
            if last_max_len < tlen:
                last_max_len = tlen
                good_el = p
        if good_el is None:
            print('no good_el')
            return ''
        print('max_len:', last_max_len)
        # 2. remove non content tag, e.g. 'script, style'
        for el in good_el.iter():
            if el.tag in self.non_content_tag or isinstance(el, HtmlComment):
                el.clear()
                el.drop_tree()
        # 3. return the main content without title
        if just_content:
            if with_tag:
                return lxml.html.tostring(good_el, encoding="utf8")
            else:
                return self.get_text(good_el)

        # 4. clean the content element's parent,
        #    which has title, author and other information
        p = good_el.getparent()
        already_has_good = False
        for el in p.iterchildren():
            if el == good_el:
                already_has_good = True
                continue
            if not el.text:
                print('drop tag:', el.tag)
                el.clear()
                el.drop_tree()
                continue
            t = el.text.strip()
            if not t or already_has_good:
                print('zero text drop tag:', el.tag)
                el.clear()
                el.drop_tree()

        if with_tag:
            return lxml.html.tostring(p, encoding="utf8")
        else:
            return self.get_text(p, is_parent=True)


if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) != 2:
        print('usage: %s html-file' % argv[0])
        exit(-1)

    f = argv[1]
    html = open(f).read()
    ext = Extractor()
    import time
    b = time.time()
    content = ext.get_content(html, just_content=True, with_tag=True)
    e = time.time()
    print ('time cost for just content: ', e-b)
    open(f+'-content.html','w').write(str(content))

    b = time.time()
    content = ext.get_content(html, just_content=False, with_tag=True)
    e = time.time()
    print ('time cost for title-plus: ', e-b)
    open(f+'-with-title.html', 'w').write(str(content))