from django.db import models


# Create your models here.
class Smartphones(models.Model):
    id = models.IntegerField(primary_key=True),
    model = models.CharField(max_length=50)
    price = models.CharField(max_length=100)
    processor = models.CharField(max_length=1000)
    ram = models.CharField(max_length=100)
    battery = models.CharField(max_length=100)
    camera = models.CharField(max_length=100)

    class Meta:
        db_table = 'Smartphones'
