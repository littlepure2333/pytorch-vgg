## 命名规范
1. 模块  
模块尽量使用`小写`命名，首字母保持小写，尽量不要用下划线(除非多个单词，且数量不多的情况)

    > 正确的模块名   
import decoder   
import html_parser   
不推荐的模块名   
import Decoder

2. 类名   
类名使用`驼峰`(CamelCase)命名风格，首字母大写，私有类可用一个下划线开头   
    > class Farm():   
      &ensp; &ensp; pass   
      class AnimalFarm(Farm):   
      &ensp; &ensp; pass   
      class _PrivateFarm(Farm):   
      &ensp; &ensp; pass   
      将相关的类和顶级函数放在同一个模块里. 不像Java, 没必要限制一个类一个模块.

3. 函数   
函数名一律`小写`，如有多个单词，用下划线隔开   
    > def run():   
      &ensp; &ensp; pass   
      def run_with_env():   
      &ensp; &ensp; pass   
      私有函数在函数前加一个下划线_   
      class Person():   
      &ensp; &ensp; def _private_func():   
      &ensp; &ensp; &ensp; &ensp; pass

4. 变量名   
变量名尽量`小写`, 如有多个单词，用下划线隔开
    ```python
    if __name__ == '__main__':
        count = 0
        school_name = ''
    # 常量采用全大写，如有多个单词，使用下划线隔开
    MAX_CLIENT = 100
    MAX_CONNECTION = 1000
    CONNECTION_TIMEOUT = 600
    ```
    
5. 常量   
常量使用以下划线分隔的`大写`命名   
    ```python
    MAX_OVERFLOW = 100
    
    Class FooBar:
    
        def foo_bar(self, print_):
            print(print_)
    ```