#
---



Modern geospatial workflows often involve **large datasets** and **computationally intensive** tasks. **CubeXpress** leverages recursion, concurrency, and parallelism to optimize these operations and handle massive image retrievals from Google Earth Engine (GEE). This section provides a concise overview of each concept and how **CubeXpress** implements them.

---

## **1. Concurrency** ⚙️

- **Definition**: Concurrency refers to managing multiple tasks such that they all make progress over the same time period—even if they’re not strictly executing at the exact same moment.  
- **Usage in CubeXpress**:  
  - The package uses **`ThreadPoolExecutor`** to coordinate concurrent downloads inside `getcube()`.  
  - By submitting multiple requests for images simultaneously, CubeXpress maximizes CPU and network usage, drastically reducing the overall download duration.

---

## **2. Parallelism** 🖥️

- **Definition**: Parallelism describes running multiple tasks *truly* at the same time, often on separate CPU cores.  
- **Usage in CubeXpress**:  
  - On multi-core systems, **thread-based parallelism** ensures that multiple image downloads or computations happen concurrently—spreading the load across available cores.  
  - This can significantly improve performance when dealing with large or numerous GEE images.

### **Concurrency vs. Parallelism**

- **Concurrency**: Tasks appear to run simultaneously but may share a single core, each making progress in turn (time-slicing).  
- **Parallelism**: Tasks literally run *at the same time* on different CPU cores.  

In practice, **CubeXpress** exploits both—concurrent scheduling of tasks and true parallel execution if multiple CPU cores are available.

---

## **3. Recursion** 🔄

- **Definition**: Recursion is the technique of breaking down a problem into smaller, more manageable subproblems, where a function calls itself until reaching a base case.  
- **Usage in CubeXpress**:  
  - When an image exceeds GEE’s size limits, **CubeXpress** automatically splits (quad-splits) it into smaller tiles.  
  - This splitting continues recursively (`max_deep_level`) until each tile is sufficiently small to download successfully.  
  - This approach ensures you can handle massive requests without manually dividing your region of interest.

---


**CubeXpress** enables **highly scalable** and **robust** geospatial workflows with Google Earth Engine. These concepts work together to optimize performance, reduce download times, and handle exceptionally large datasets, giving you a seamless experience when building geospatial data cubes.

