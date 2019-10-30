int test(int from, int to)
{
  int i;
  int s = 0;
 
  for (i = from; i < to; i++)
    if (i % 3 == 0)
      s += i;

  return s;
}
