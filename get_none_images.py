from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
  feeder_threads=1,
  parser_threads=2,
  downloader_threads=4,
  storage={'root_dir': 'dataset/porto-dataset/images/none'})

google_crawler.crawl(keyword='monuments', max_num=300, file_idx_offset=0)