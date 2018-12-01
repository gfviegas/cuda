x = input("Entre com um texto: ")
a = e = i = o = u = 0
for y in range(0, len(x)):
    if x[y] == 'a':
        a = a + 1
    elif x[y] == 'e':
        e = e + 1
    elif x[y] == 'i':
        i = i + 1
    elif x[y] == 'o':
        o = o + 1
    elif x[y] == 'u':
        u = u + 1

print("Ocorrências das vogais:")
print("a: %d" % a)
print("e: %d" % e)
print("i: %d" % i)
print("o: %d" % o)
print("u: %d" % u)
