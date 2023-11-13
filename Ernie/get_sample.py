import random

def get_product(**kwargs):
    skin_type = kwargs.get("skin_type")
    season = kwargs.get("season")
    effect = kwargs.get("effect")
    category = kwargs.get("category")
    brand = kwargs.get("brand")
    item_desc = kwargs.get("item_desc")
    shop = kwargs.get("shop")
    index = kwargs.get("index")
    num = kwargs.get("num")
    specs = kwargs.get("specs")
    taste = kwargs.get("taste")
    date = kwargs.get("date")
    activity = kwargs.get("activity")
    customer = kwargs.get("customer")
    age = kwargs.get("age")
    color = kwargs.get("color")
    made_in = kwargs.get("made_in")
    gender = kwargs.get("gender")


    skin_type = random.choice(skin_type)
    season = random.choice(season)
    effect = random.choice(effect)
    category = random.choice(category)
    brand = random.choice(brand)
    item_desc = random.choice(item_desc)
    shop = random.choice(shop)
    index = random.choice(index)
    num = random.choice(num)
    specs = random.choice(specs)
    taste = random.choice(taste)
    date = random.choice(date)
    activity = random.choice(activity)
    customer = random.choice(customer)
    age = random.choice(age)
    color = random.choice(color)
    made_in = random.choice(made_in)

    data = [
        {
            "text": f"帮我推荐一款{skin_type}{season}使用的{effect}的{category}。",
            "output": {"skin_type": skin_type, "season": season, "effect": effect, 
                    "category": category, "target": "product", "intent_1": "商品"}
        },
        {
            "text": f"{brand}{taste}可以搭配什么{category}去用？",
            "output": {"brand": brand, "taste": taste, "category": category, 
                    "target": "product", "intent_1": "商品"}
        },
        {
            "text": f"{shop}{index}排名前{num}的{category}是哪{num}款？",
            "output": {"shop": shop, "index": index, "top": num, "category": category, 
                    "target": "product", "intent_1": "商品"}
        },
        {
            "text": f"{shop}容量是{specs}的{category}有哪些？",
            "output": {"shop": shop, "specs": specs, "category": category, 
                    "target": "product", "intent_1": "商品"}
        },
        {
            "text": f"有没有类似{brand}{taste}这种香味的{effect}{category}？",
            "output": {"brand": brand, "taste": taste, "effect": effect, "category": category, 
                    "target": "product", "intent_1": "商品"}
        },
        {
            "text": f"{date}{index}最多的{category}是哪一款？",
            "output": {"date": date, "index": index,  "category": category, 
                    "target": "product", "intent_1": "商品"}
        },
        {
            "text": f"{date}大家都在推荐什么{activity}？",
            "output": {"date": date, "activity": activity, 
                    "target": "product", "intent_1": "商品"}
        },
        {
            "text": f"{brand}{item_desc}{customer}价是多少？",
            "output": {"brand": brand, "item_desc": item_desc, "customer": customer, 
                    "target": "price", "intent_1": "商品"}
        },
        {
            "text": f"{brand}{category}主要功效是什么？",
            "output": {"brand": brand, "category": category, "target": "effect", "intent_1": "商品"}
        }
    ]
    return data

def get_knowledge(**kwargs):
    brand = kwargs.get("brand")
    category = kwargs.get("category")
    skin_type = kwargs.get("skin_type")
    item_desc = kwargs.get("item_desc")
    season = kwargs.get("season")
    effect = kwargs.get("effect")
    star = kwargs.get("star")
    age = kwargs.get("age")
    component = kwargs.get("component")
    pain = kwargs.get("pain")
    texture = kwargs.get("texture")
    net_content = kwargs.get("net_content")
    gender = kwargs.get("gender")
    made_in = kwargs.get("made_in")
    shop = kwargs.get("shop")
    price = kwargs.get("price")
    color = kwargs.get("color")
    

    brand1 = random.choice(brand)
    brand2 = random.choice(brand)
    category = random.choice(category)
    skin_type = random.choice(skin_type)
    item_desc = random.choice(item_desc)
    season = random.choice(season)
    effect = random.choice(effect)
    star = random.choice(star)
    age = random.choice(age)
    component = random.choice(component)
    pain = random.choice(pain)
    texture = random.choice(texture)
    net_content = random.choice(net_content)
    gender = random.choice(gender)
    made_in = random.choice(made_in)
    shop = random.choice(shop)
    price = random.choice(price)
    color = random.choice(color)

    data = [
        {
            "text": f"{brand1}{category}和{brand2}{category}相比哪一款更适合{skin_type}？",
            "output": {"brand": f"{brand1},{brand2}", "category": category, "skin_type": skin_type, 
                    "intent_1": "内容"}
        },
        {
            "text": f"{brand1}{category}的主要成分是什么？具体有什么作用？",
            "output": {"brand": f"{brand1}", "category": category,
                    "intent_1": "内容"}
        },
        {
            "text": f"我买的{brand2}{item_desc}怎么用，有没有操作指南？",
            "output": {"brand": f"{brand2}", "item_desc": item_desc,
                    "intent_1": "内容"}
        },
        {
            "text": f"{brand1}{item_desc}的用户评价怎么样？",
            "output": {"brand": brand1, "item_desc": item_desc,
                    "intent_1": "内容"}
        },
        {
            "text": f"{season}用什么颜色的{category}才能更{effect}啊？",
            "output": {"season": season, "category": category, "effect": effect, 
                     "intent_1": "内容"}
        },
        {
            "text": f"有没有{brand1}{item_desc}的测评内容可以参考一下？",
            "output": {"brand1": brand1, "item_desc": item_desc,
                     "intent_1": "内容"}
        },
        {
            "text": f"{star}代言的{category}是什么？",
            "output": {"star": star, "category": category,
                     "intent_1": "内容"}
        },
        {
            "text": f"{age}{skin_type}{effect}应该怎么去护肤？",
            "output": {"age": age, "skin_type": skin_type, "effect": effect,
                     "intent_1": "内容"}
        },
        {
            "text": f"{component}真的可以{effect}吗？原理是什么？",
            "output": {"component": component, "effect": effect,
                     "intent_1": "内容"}
        },
        {
            "text": f"{skin_type}定妆用散粉还是粉饼更合适？",
            "output": {"skin_type": skin_type,
                     "intent_1": "内容"}
        },
        {
            "text": f"我最近{pain}，想要{effect}，可以吃点什么{category}吗？",
            "output": {"pain": pain, "effect": effect, "category": category,
                     "intent_1": "内容"}
        },
        {
        "text": f"我想了解{brand}品牌的{item_desc}，价格多少？",
        "output": {
            "brand": brand,
            "item_desc": item_desc,
            "intent_1": "内容"
        }
    },
    {
        "text": f"{brand}的{category}适合{season}使用吗？",
        "output": {
            "brand": brand,
            "category": category,
            "season": season,
            "intent_1": "内容"
        }
    },
    {
        "text": f"对于{age}岁的人，你们有推荐的{category}产品吗？",
        "output": {
            "age": age,
            "category": category,
            "intent_1": "内容"
        }
    },
    {
        "text": f"这款{texture}的{item_desc}怎么样？",
        "output": {
            "texture": texture,
            "item_desc": item_desc,
            "intent_1": "内容"
        }
    },
    {
        "text": f"有哪些{net_content}装的{category}？",
        "output": {
            "net_content": net_content,
            "category": category,
            "intent_1": "内容"
        }
    },
    {
        "text": f"{gender}使用{brand}的{item_desc}效果好吗？",
        "output": {
            "gender": gender,
            "brand": brand,
            "item_desc": item_desc,
            "intent_1": "内容"
        }
    },
    {
        "text": f"你们{shop}有没有{brand}的{item_desc}卖？",
        "output": {
            "shop": shop,
            "brand": brand,
            "item_desc": item_desc,
            "intent_1": "内容"
        }
    },
    {
        "text": f"{made_in}的{category}质量怎么样？",
        "output": {
            "made_in": made_in,
            "category": category,
            "intent_1": "内容"
        }
    },
    {
        "text": f"{price}元以下有没有好用的{category}？",
        "output": {
            "price": price,
            "category": category,
            "intent_1": "内容"
        }
    },
    {
        "text": f"{color}色的{item_desc}适合{season}季节吗？",
        "output": {
            "color": color,
            "item_desc": item_desc,
            "season": season,
            "intent_1": "内容"
        }
    }
    ]
    return data

def get_activity(**kwargs):

    brand = kwargs.get("brand")
    category = kwargs.get("category")
    skin_type = kwargs.get("skin_type")
    item_desc = kwargs.get("item_desc")
    season = kwargs.get("season")
    effect = kwargs.get("effect")
    star = kwargs.get("star")
    age = kwargs.get("age")
    component = kwargs.get("component")
    pain = kwargs.get("pain")
    activity = kwargs.get("activity")
    shop = kwargs.get("shop")
    customer = kwargs.get("customer")
    date = kwargs.get("date")
    city = kwargs.get("city")
    activity = kwargs.get("activity")


    brand1 = random.choice(brand)
    brand2 = random.choice(brand)
    category = random.choice(category)
    skin_type = random.choice(skin_type)
    item_desc = random.choice(item_desc)
    season = random.choice(season)
    effect = random.choice(effect)
    star = random.choice(star)
    age = random.choice(age)
    component = random.choice(component)
    pain = random.choice(pain)
    activity = random.choice(activity)
    shop = random.choice(shop)
    customer = random.choice(customer)
    date = random.choice(date)
    city = random.choice(city)
    activity = random.choice(activity)

    data = [
        {
            "text": f"{brand1}{category}有什么优惠吗？",
            "output": {"brand": brand1, "category": category, "intent_1": "活动"}
        },
        {
            "text": f"{brand2}{item_desc}{activity}的福利有什么？",
            "output": {"brand": brand2, "item_desc": item_desc, "activity": activity, "intent_1": "活动"}
        },
        {
            "text": f"{activity}店内有什么300以内的满减优惠吗",
            "output": {"activity": activity, "price": 300, "lower": "以内", "intent_1": "活动"}
        },
        {
            "text": f"最近{shop}有什么{customer}折扣吗？",
            "output": {"shop": shop, "customer": customer, "intent_1": "活动"}
        },
        {
            "text": f"{activity}{category}产品有什么促销活动吗？",
            "output": {"activity": activity, "category": category, "activity": "促销活动", "intent_1": "活动"}
        },
        {
            "text": f"{customer}专享有哪些可以领取的优惠券？",
            "output": {"customer": customer, "intent_1": "活动"}
        },
        {
            "text": f"{date}{city}门店有什么{activity}吗？",
            "output": {"date": date, "city": city, "activity": activity, "intent_1": "活动"}
        },
        {
            "text": f"{activity}有哪些奖品可以领？",
            "output": {"activity": activity, "intent_1": "活动"}
        },
        {
            "text": f"{brand1}的新品{category}有没有{activity}？怎么申请？",
            "output": {"brand": brand1, "category": category, "activity": activity, "intent_1": "活动"}
        },
        {
        "text": f"最近{brand1}有没有{season}季的{category}优惠？",
        "output": {
            "brand": brand1,
            "season": season,
            "category": category,
            "intent_1": "活动"
        }
    },
    {
        "text": f"{brand1}的{item_desc}参加{activity}是什么条件？",
        "output": {
            "brand": brand1,
            "item_desc": item_desc,
            "activity": activity,
            "intent_1": "活动"
        }
    },
    {
        "text": f"{category}现在有什么{activity}吗？",
        "output": {
            "category": category,
            "activity": activity,
            "intent_1": "活动"
        }
    },
    {
        "text": f"{brand1}今年{date}的优惠活动是什么？",
        "output": {
            "brand": brand1,
            "date": date,
            "intent_1": "活动"
        }
    },
    {
        "text": f"{activity}时{brand1}{category}的{item_desc}有什么特别优惠？",
        "output": {
            "activity": activity,
            "brand": brand1,
            "category": category,
            "item_desc": item_desc,
            "intent_1": "活动"
        }
    }
    ]
    return data

def get_servicer(**kwargs):
    brand = kwargs.get("brand")
    category = kwargs.get("category")
    skin_type = kwargs.get("skin_type")
    item_desc = kwargs.get("item_desc")
    season = kwargs.get("season")
    effect = kwargs.get("effect")
    star = kwargs.get("star")
    age = kwargs.get("age")
    component = kwargs.get("component")
    pain = kwargs.get("pain")
    activity = kwargs.get("activity")
    shop = kwargs.get("shop")
    customer = kwargs.get("customer")
    date = kwargs.get("date")
    city = kwargs.get("city")
    activity = kwargs.get("activity")
    complaint_question = kwargs.get("complaint_question")
    complaint_plan = kwargs.get("complaint_plan")
    texture = kwargs.get("texture")


    brand1 = random.choice(brand)
    brand2 = random.choice(brand)
    category = random.choice(category)
    skin_type = random.choice(skin_type)
    item_desc = random.choice(item_desc)
    season = random.choice(season)
    effect = random.choice(effect)
    star = random.choice(star)
    age = random.choice(age)
    component = random.choice(component)
    pain = random.choice(pain)
    activity = random.choice(activity)
    shop = random.choice(shop)
    customer = random.choice(customer)
    date = random.choice(date)
    city = random.choice(city)
    activity = random.choice(activity)
    complaint_question = random.choice(complaint_question)
    complaint_plan = random.choice(complaint_plan)
    texture = random.choice(texture)
    symptom = random.choice(symptom)

    data = [
        {
            "text": f"我购买的{brand1}{category}{complaint_question}，麻烦帮我{complaint_plan}。",
            "output": {"brand": brand1, "category": category, "complaint_question": complaint_question, 
                     "complaint_plan": complaint_plan, "intent_1": "客诉"}
        },
        {
            "text": f"我下单的{brand2}{effect}{category}{complaint_question}了，你能帮我{complaint_plan}的吗？",
            "output": {"brand": brand2, "category": category, "complaint_question": complaint_question, 
                     "complaint_plan": complaint_plan, "intent_1": "客诉"}
        },
        {
            "text": f"我买的{brand1}{category}{complaint_question}，可以帮我{complaint_plan}吗？",
            "output": {"brand": brand1, "category": category, "complaint_question": complaint_question, 
                     "complaint_plan": complaint_plan, "intent_1": "客诉"}
        },
        {
            "text": f"我买的{brand2}{category}和之前用的不一样，{complaint_question}，给我{complaint_plan}。",
            "output": {"brand": brand2, "category": category, "complaint_question": complaint_question, 
                     "complaint_plan": complaint_plan, "intent_1": "客诉"}
        },
        {
            "text": f"我买的{brand1}{category}怎么{complaint_question}，给我{complaint_plan}！",
            "output": {"brand": brand1, "category": category, "complaint_question": complaint_question, 
                     "complaint_plan": complaint_plan, "intent_1": "客诉"}
        },
        {
            "text": f"我的{complaint_question}，能不能帮我{complaint_plan}。",
            "output": {"complaint_question": complaint_question, 
                     "complaint_plan": complaint_plan, "intent_1": "客诉"}
        },
        {
            "text": f"我用你们{category}的{item_desc}{complaint_question}了，马上给我{complaint_plan}！",
            "output": {"category": category, "item_desc": item_desc, "complaint_question": complaint_question,
                     "complaint_plan": complaint_plan, "intent_1": "客诉"}
        },
        {
            "text": f"我买的{brand1}{category}已经确认收货了，能不能{complaint_question}，要怎么开。",
            "output": {"brand1": brand1, "category": category, "complaint_question": complaint_question,
                     "intent_1": "客诉"}
        },
        {
        "text": f"我刚买的{brand}{item_desc}有问题，{complaint_question}，请问如何{complaint_plan}？",
        "output": {
            "brand": brand,
            "item_desc": item_desc,
            "complaint_question": complaint_question,
            "complaint_plan": complaint_plan,
            "intent_1": "客诉"
        }
    },
    {
        "text": f"我订的{brand}{item_desc}使用后出现{symptom}，怎么办？",
        "output": {
            "brand": brand,
            "item_desc": item_desc,
            "symptom": symptom,
            "intent_1": "客诉"
        }
    },
    {
        "text": f"我买的{item_desc}说是{made_in}的，但实际不是, {complaint_question}",
        "output": {
            "item_desc": item_desc,
            "made_in": made_in,
            "complaint_question": complaint_question,
            "intent_1": "客诉"
        }
    },
    {
        "text": f"为什么我购买的{brand}{item_desc}{complaint_question}，{complaint_plan}？",
        "output": {
            "brand": brand,
            "item_desc": item_desc,
            "complaint_question": complaint_question,
            "complaint_plan":complaint_plan,
            "intent_1": "客诉"
        }
    },
    {
        "text": f"你们店铺的{item_desc}{complaint_question}，能否帮忙{complaint_plan}？",
        "output": {
            "item_desc": item_desc,
            "complaint_question": complaint_question,
            "complaint_plan": complaint_plan,
            "intent_1": "客诉"
        }
    },
    {
        "text": f"{brand}{item_desc}的{texture}感觉不对，{complaint_question}。",
        "output": {
            "brand": brand,
            "item_desc": item_desc,
            "texture": texture,
            "complaint_question": complaint_question,
            "intent_1": "客诉"
        }
    },
    {
        "text": f"我发现你们{shop}卖的{item_desc}{made_in}{complaint_question}",
        "output": {
            "shop": shop,
            "item_desc": item_desc,
            "made_in": made_in,
            "complaint_question": complaint_question,
            "intent_1": "客诉"
        }
    },
    {
        "text": f"我买的{item_desc}{evaluate}，{complaint_plan}？",
        "output": {
            "item_desc": item_desc,
            "evaluate": evaluate,
            "complaint_plan": complaint_plan,
            "intent_1": "客诉"
        }
    }
    ]
    return data


